#coding:utf-8

import sqlite3 as sqlite
from datetime import datetime
from DBConstants import DBConstants

# 指定したテーブルに書き込み、読み込みを行うヘルパクラス
# テーブル自体は既にあるものとする
class DBHelper:

    DB_PATH  = None
    TABLE_NAME = None
    TABLE_COLUMNS = None

    # ヘルパで扱うテーブル名とカラムを指定
    def __init__(self, dbPass, tableName, tableColumns):

        self.DB_PATH = dbPass
        self.TABLE_NAME = tableName
        self.TABLE_COLUMNS = tableColumns

        # コネクションを作成。とりあえず分離レベルはデフォルトとして問題があったら修正
        self.connection = sqlite.connect(self.DB_PATH) #, isolation_level='EXCLUSIVE')
        self.cursor = self.connection.cursor()

    # テーブル作成用。基本使わない
    def createTable(self, createTableString):

        sqlStr = "CREATE TABLE " + self.TABLE_NAME + " ("+ createTableString+");"
        print(sqlStr)
        self.cursor.execute(sqlStr)

    # テーブルに追加するデータをリストで指定
    def add(self, data):
        if data is None or len(data) ==0:
            return

        # データ追加用のSQL文の作成
        sqlStr = "INSERT INTO " + self.TABLE_NAME + " ("+self.TABLE_COLUMNS+") VALUES('"
        for i in range(len(data)-1):
            sqlStr = sqlStr + str(data[i]) + "', '"

        sqlStr =  sqlStr + str(data[len(data)-1]) + "');"

#        print("Execute:")
#        print(sqlStr)

        self.cursor.execute(sqlStr)

    # 指定した時間内のデータをリストにして返す
    def get(self, startTime, endTime):

        timeStr = time.strftime(DBConstants.TIME_FORMAT)
        sqlStr = "SELECT " + self.TABLE_COLUMNS + \
        " FROM " + self.TABLE_NAME + \
        " WHERE " + DBConstants.TIME_COLUMN + " >= " + startTime + \
        " AND " + DBConstants.TIME_COLUMN + " <= " + endTime + \
        ";"

        self.cursor.execute(sqlStr)

        return self.cursor.fetchall()

    # 全てのデータのCursorを返す
    def getAllDataCursor(self):

        sqlStr = "SELECT " + self.TABLE_COLUMNS + \
        " FROM " + self.TABLE_NAME + ";"

        allCursor = self.connection.cursor()
        allCursor.execute(sqlStr)
        return allCursor

    # 指定した時刻以降のCursorを返す
    def getCursor(self, time=None):

        # 時刻が指定されていない場合は最初からとする
        if time is None:
            time = datetime.utcfromtimestamp(0)

        if isinstance(time, str) == True:
            timeStr = time
        else:
            timeStr = time.strftime(DBConstants.TIME_FORMAT)

        sqlStr = "SELECT " + self.TABLE_COLUMNS + \
        " FROM " + self.TABLE_NAME + \
        " WHERE " + DBConstants.TIME_COLUMN + " >= '" + timeStr + "'" + \
        ";"

        tmpCursor = self.connection.cursor()
        tmpCursor.execute(sqlStr)
        return tmpCursor

    # valueは数値型とする
    def insertOrUpdate(self, time, columnName, value):

        timeStr = time.strftime(DBConstants.TIME_FORMAT)

        # とりあえずUPDATEしてみる
        sqlStr = "UPDATE " + self.TABLE_NAME + " SET " + columnName + " = " + str(value) + " WHERE time = '" + timeStr + "';"
        self.cursor.execute(sqlStr)
 #       print("Execute:")
 #       print(sqlStr)

        # UPDATE件数が0なら、INSERTする
        if self.cursor.rowcount == 0:
            sqlStr = "INSERT INTO " + self.TABLE_NAME + " (time,"+columnName+") VALUES('" + timeStr + "'," + str(value) + ");"
            self.cursor.execute(sqlStr)

#            print("Execute:")
#            print(sqlStr)

    def commit(self):
        self.connection.commit()

    def close(self):
        self.cursor.close()

