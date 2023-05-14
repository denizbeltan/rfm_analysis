import pandas as pd
import datetime as dt
import plotly.express as px
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from flask import Flask, send_file
import io

application = Flask(__name__)

def RClass(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4


def FMClass(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


def RFM():
    ###############################

    # sutun isimleri
    # DocumentID,Date,SKU,Price,Discount,Customer,Quantity

    ###############################

    # csv dosyasini ice aktarma
    df = pd.read_csv('file_out.csv')
    df.head()

    # ise yaramayan sutunu kaldirma(ilk sutun)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.head()

    # veri setinin baslangic ve bitis tarihlerini kontrol etme
    df['Date'].min(), df['Date'].max()

    # urun, islem ve musterinin toplam sayisi

    pd.DataFrame([{'products': len(df['SKU'].value_counts()),
                   'transactions': len(df['DocumentID'].value_counts()),
                   'customers': len(df['Customer'].value_counts()),
                   }], columns=['products', 'transactions', 'customers'], index=['quantity'])

    # veri formatını değiştirme
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    #############################
    # Recency Data Frame#

    # musterilerin herhangi bir urunu satin aldigi son tarihe gore gruplama
    recency_df = df.groupby(['Customer'], as_index=False)['Date'].max()
    recency_df.columns = ['CustomerID', 'LastPurchaseDate']
    recency_df.head()

    # verilerimizdeki islemlerde en son tarihi guncel tarih olarak kullanmak
    now = df['Date'].max()

    # Recency(Yenilik) hesapla (musteri urunu en son tarihe (2022-11-09) gore kac gun once satin aldi))
    recency_df['Recency'] = recency_df.LastPurchaseDate.apply(lambda x: (now - x).days)
    recency_df.head()

    # son alım tarihi sütununa artık ihtiyacımız yok
    recency_df.drop(columns=['LastPurchaseDate'], inplace=True)
    recency_df.head()

    # Frequency Data Frame#

    # musteriler tarafindan kac islem yapildigina bagli olarak siklik DF'si olusturma
    frequency_df = df.copy()
    frequency_df.drop_duplicates(subset=['Customer', 'DocumentID'], keep="first", inplace=True)
    frequency_df = frequency_df.groupby('Customer', as_index=False)['DocumentID'].count()
    frequency_df.columns = ['CustomerID', 'Frequency']
    frequency_df.head()

    # Monetary Data Frame#

    # bir muşterinin her islemde ne kadar harcadigini hesaplama
    df['Total_cost'] = df['Price'] * df['Quantity']
    df.head()

    # musterinin son tarihe gore ozet harcamasini kontrol etme
    monetary_df = df.groupby('Customer', as_index=False)['Total_cost'].sum()
    monetary_df.columns = ['CustomerID', 'Monetary']
    monetary_df.head()

    # uc data framei tek bir df'de birlestirelim

    # recency ve frequency birlestirelim
    rf = recency_df.merge(frequency_df, left_on='CustomerID', right_on='CustomerID')

    # rf frame'ini monetary değeri ile birlestirelim

    rfm = rf.merge(monetary_df, left_on='CustomerID', right_on='CustomerID')

    rfm.set_index('CustomerID', inplace=True)
    rfm = rfm[rfm['Monetary'] > 0]
    rfm.head()

    ######################
    # RFM hesaplanmasi
    #######################
    # lifetimes kutuphanesini ice aktarma
    # argumanlar (x = value, p = recency, monetary_value, frequency, k = quartiles dict(0.25,0.50,0.75))

    grp_df = df.groupby('Customer').Total_cost.sum().sort_values(ascending=False)

    lf_data = summary_data_from_transaction_data(df, 'Customer', 'Date', monetary_value_col='Total_cost',
                                                 observation_period_end=now)

    lf_data['frequency'] = lf_data['frequency'].astype(int)
    lf_data['recency'] = lf_data['recency'].astype(int)
    lf_data['T'] = lf_data['T'].astype(int)
    lf_data['monetary_value'] = lf_data['monetary_value'].astype(int)

    lf_data.head()

    quantiles = lf_data.quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()

    rfmSegmentation = lf_data

    rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency', quantiles,))
    rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency', quantiles,))
    rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass,
                                                                            args=('monetary_value', quantiles,))

    # classları etiketleme
    rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str) \
                                  + rfmSegmentation.F_Quartile.map(str) \
                                  + rfmSegmentation.M_Quartile.map(str)

    # r' ile regular expression kullanildi. yani recency degeri icin soldaki [] araligina bakacakken frequency icin
    # sagdaki [] araligina bakilacak.
    # gercek RFM classlarindan olusturdugumuz classlara atama:
    segt_map = {
        r'[1-2][1-2]': 'Hibernate',
        r'[1-2][3-4]': 'Risk',
        r'[1-2]5': 'Cant Lose',
        r'3[1-2]': 'Sleeper',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal',
        r'41': 'Promising',
        r'51': 'New Customer',
        r'[4-5][2-3]': 'High Potential',
        r'5[4-5]': 'Champion'
    }

    rfmSegmentation['Segment'] = rfmSegmentation['R_Quartile'].map(str) + rfmSegmentation['F_Quartile'].map(str)
    rfmSegmentation['Segment'] = rfmSegmentation['Segment'].replace(segt_map, regex=True)

    rfmSegmentation.head()
    rfmSegmentation['Segment'].value_counts()
    rfmSegmentation.sort_values('monetary_value')

    df_clv = rfmSegmentation

    # lifetimes kutuphanesinden BetaGeoFitteri ice aktarma
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(df_clv['frequency'], df_clv['recency'], df_clv['T'])

    # lifetimes kutuphanesinden GammaGammaFitteri ice aktarma
    rfmSegmentation2 = rfmSegmentation[rfmSegmentation['monetary_value'] > 0]
    rfmSegmentation1 = rfmSegmentation[rfmSegmentation['monetary_value'] == 0]
    rfmSegmentation1['CLV'] = 0

    GGF = GammaGammaFitter(penalizer_coef=0)

    GGF.fit(rfmSegmentation2["frequency"], rfmSegmentation2["monetary_value"])

    rfmSegmentation2["CLV"] = GGF.customer_lifetime_value(bgf, rfmSegmentation2["frequency"],
                                                          rfmSegmentation2["recency"],
                                                          rfmSegmentation2["T"], rfmSegmentation2["monetary_value"],
                                                          time=12,
                                                          discount_rate=0.01, freq="D")
    rfm_Seg = pd.concat([rfmSegmentation1, rfmSegmentation2])
    rfm_Seg.head()
    print(rfmSegmentation)

    rfm_Seg.sort_values('CLV', ascending=False)

    # plotly kutuphanesini ice aktarma
    fig = px.treemap(rfm_Seg, path=['Segment'], values='M_Quartile',
                     color_discrete_sequence=["maroon", "olive", "teal", "peru", "indianred", "orangered"])
    fig.update_traces(root_color="white")
    fig.update_layout(margin=dict(t=50, l=50, r=50, b=50))

    #fig.show()

with open("static/newplot.png", "rb") as f:
    image = f.read()

@application.route('/')
def dash_show():
    RFM()
    return send_file(
        io.BytesIO(image),
        mimetype="image/png",
        as_attachment=False
    )

if __name__ == '__main__':
    application.run()

