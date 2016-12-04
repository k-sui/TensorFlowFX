#coding:utf-8

class DBConstants(object):

    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    TIME_COLUMN = "time"

    CHART_1MIN_TBL = "chart1min"
    CHART_5MIN_TBL = "chart5min"
    CHART_10MIN_TBL = "chart10min"
    CHART_30MIN_TBL = "chart30min"
    CHART_1HOUR_TBL = "chart1hour"
    CHART_4HOUR_TBL = "chart4hour"
    CHART_1DAY_TBL = "chart1day"
    CHART_COLUMNS = "time, firstPrice, highPrice, lowPrice, endPrice"

    INDICATOR_1MIN_TBL = "indicator1min"
    INDICATOR_5MIN_TBL = "indicator5min"
    INDICATOR_10MIN_TBL = "indicator10min"
    INDICATOR_30MIN_TBL = "indicator30min"
    INDICATOR_1HOUR_TBL = "indicator1hour"
    INDICATOR_1HOUR_FORMAT_TBL = "indicator1hour_formated"
    INDICATOR_4HOUR_TBL = "indicator4hour"
    INDICATOR_1DAY_TBL = "indicator1day"
    INDICATOR_COLUMNS = "time, sma5, sma25, rsi9, rsi11, rsi14, macd, macdsignal, macdhist, cci14, cci20, stocSlowk, stocSlowd, stocFastk, stocFastd, stocRSIk, stocRSId, trima5, trima25 "

    RESULT_1MIN_TBL = "result1min"
    RESULT_5MIN_TBL = "result5min"
    RESULT_10MIN_TBL = "result10min"
    RESULT_30MIN_TBL = "result30min"
    RESULT_1HOUR_TBL = "result1hour"
    RESULT_4HOUR_TBL = "result4hour"
    RESULT_1DAY_TBL = "result1day"
    RESULT_COLUMNS = "time, pro05_loss01"

    # 取引結果のDB上での定義
    RESULT_BUY = 1
    RESULT_SELL = -1
    RESULT_NOR = 0

    CREATE_INDICATOR_TABLE_STRING = "id integer PRIMARY KEY AUTOINCREMENT, time text unique, " \
                                  + "sma5 float, sma25 float, " \
                                  + "rsi9 float, rsi11 float, rsi14 float, " \
                                  + "macd float, macdsignal float, macdhist float, " \
                                  + "cci14 float, cci20 float, " \
                                  + "stocSlowk float, stocSlowd float, " \
                                  + "stocFastk float, stocFastd float, " \
                                  + "stocRSIk float, stocRSId float, " \
                                  + "trima5 float, trima25 float "

    CREATE_CHART_TABLE_STRING = "id integer PRIMARY KEY AUTOINCREMENT, time text unique, firstPrice float, highPrice float, lowPrice float, endPrice float"

    CREATE_RESULT_TABLE_STRING = "id integer PRIMARY KEY AUTOINCREMENT, time text unique, pro05_loss01 integer"