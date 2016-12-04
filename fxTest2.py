import tensorflow as tf
import csv
import random

from DBConstants import  DBConstants
from DBHelper import DBHelper

#csvファイル内の列の位置を示す定数
TIME = 0
OPEN_PRICE = 1
HIGH_PRICE = 2
LOW_PRICE = 3
CLOSE_PRICE = 4
INDI_FIRST = 1  # インジケータが入る最初の列
INDI_LAST = 19  # インジケータが入る最後の列

# 金額の上下を決めるための閾値を割合で指定
DIF_RATA_THRESHOLD = 0.0003

# 隠れ層の数
HIDDEN_LAYER_NUM_1 = 20
HIDDEN_LAYER_NUM_2 = 30
HIDDEN_LAYER_NUM_3 = 15
HIDDEN_LAYER_NUM_4 = 15
HIDDEN_LAYER_NUM_5 = 10
HIDDEN_LAYER_NUM_6 = 5


DB_PATH = "/home/k-sui/fxdata/fxdata.db"
XMIN = 60

def inference(x_placeholder):
#def inference(x_placeholder, keep_prob):

    with tf.name_scope('hidden1'):
        # 始値からインジケータの最後までを使って学習するので、入力はINDI_LAST個
        weights = tf.Variable(tf.truncated_normal([INDI_LAST-1, HIDDEN_LAYER_NUM_1], stddev=1.0), name='weights')  # stddevはランダム値で初期化するときの標準偏差
        biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_1]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(x_placeholder, weights) + biases)    # tf.matmul : 行列の積の計算


    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_1, HIDDEN_LAYER_NUM_2], stddev=1.0), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_2]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('hidden3'):
        weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_2, HIDDEN_LAYER_NUM_3], stddev=1.0), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_3]), name='biases')
        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

    # with tf.name_scope('hidden4'):
    #     weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_3, HIDDEN_LAYER_NUM_4], stddev=1.0),
    #                           name='weights')
    #     biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_4]), name='biases')
    #     hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)
    #
    # with tf.name_scope('hidden5'):
    #     weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_4, HIDDEN_LAYER_NUM_5], stddev=1.0),
    #                           name='weights')
    #     biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_5]), name='biases')
    #     hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)
    #
    # with tf.name_scope('hidden6'):
    #     weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_5, HIDDEN_LAYER_NUM_6], stddev=1.0),
    #                           name='weights')
    #     biases = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM_6]), name='biases')
    #     hidden6 = tf.nn.relu(tf.matmul(hidden5, weights) + biases)

            #DropOut
#    dropout = tf.nn.dropout(hidden2, keep_prob)

    with tf.name_scope('softmax'):
        weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_3, 3], stddev=1.0), name='weights')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        y = tf.nn.softmax(tf.matmul(hidden3, weights) + biases)
#        y = tf.nn.softmax(tf.matmul(dropout, weights) + biases)

    return y

def loss(y, target_placeholder):
    # yに0が入るとおかしくなるので、1e-10〜1.0の範囲に指定(http://qiita.com/ikki8412/items/3846697668fc37e3b7e0)
    return -tf.reduce_sum(target_placeholder * tf.log(tf.clip_by_value(y,1e-10,1.0)))

def optimize(loss, l_rate=0.001):
    optimizer = tf.train.AdamOptimizer(learning_rate = l_rate)
    train_step = optimizer.minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数

    引数:
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      accuracy: 正解率(float)

    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy


def readData():

    # DBから読み込んだインジケータを保持する配列
    x_data = []

    # DBから読み込んだ正解データを保持する配列
    y_ans = []

    # DBHelperの作成
    indicatorHelper = createIndicatorXminHelper(DB_PATH, XMIN)
    resultHelper = createResultXminHelper(DB_PATH, XMIN)

    # 価格上下データの分布の確認用
    buy = 0
    sell = 0
    wait = 0

    # 時刻部分を除く前データを取得
    for row in indicatorHelper.getAllDataCursor():
        x_data.append(list(row)[1:])
        if len(x_data)  == 96000:
            print("time check x:")
            print(row)

    for row in resultHelper.getAllDataCursor():
        if row[1] == 1:
            y_ans.append([1.,0.,0.])
            buy += 1
        elif row[1] == -1:
            y_ans.append([0.,0.,1.])
            sell += 1
        else:
            y_ans.append([0.,1.,0.])
            wait += 1
        if len(y_ans) == 96000 or len(y_ans) == 96900:
            print("time check y:")
            print(row)

    print("[x, y] : ")
    print([len(x_data), len(y_ans)])

    print("[buy, sell, wait] : ")
    print([buy, sell, wait])

    #
    return x_data[50:], y_ans[50:]

# 指定した比率でランダムに学習データとテストデータに分割する
# testRatio = 0.0 〜 1.0
def splitData(x, y, testRatio):

    # 学習用データと評価用データを分割
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # デバッグ用。
    print("len(x)=" + str(len(x)) + "  len(y)=" + str(len(y)))

    # 最後のデータには教師データ(y)が無いので、最後のデータは除く
    for i in range(0, len(x)-1):


        # midのデータが多いので、buy,sellに合わせて削る
#        rnd = random.random()
#        if y[i][1] == 1:
#            if rnd>=0.27:
#                continue

        rnd = random.random()
        if rnd >= testRatio:
            # midのデータが多いので、buy,sellに合わせて削る
            rnd = random.random()
            if y[i][1] == 1:
                if rnd >= 0.27:
                    continue

            x_train.append(x[i])
            y_train.append(y[i])

        else:
            x_test.append(x[i])
            y_test.append(y[i])

    buy = 0
    sell = 0
    mid = 0

    # for i in range(0, len(y_test) - 1):
    #     if y_test[i][0] == 1:
    #         buy += 1
    #     elif y_test[i][1] == 1:
    #         mid += 1
    #     else:
    #         sell += 1

    for i in range(0, len(y_train) - 1):
        if y_train[i][0] == 1:
            buy += 1
        elif y_train[i][1] == 1:
            mid += 1
        else:
            sell += 1

    print("[buy, mid, sell]")
    print([buy,mid,sell])


    return x_train, y_train, x_test, y_test

def divideData(x,y):

    x_buy = []
    x_mid = []
    x_sell = []

    y_buy = []
    y_mid = []
    y_sell = []

    for i in range(0, len(y) - 1):
        if y[i][0] == 1:
            x_buy.append(x[i])
            y_buy.append(y[i])

        elif y[i][1] == 1:
            x_mid.append(x[i])
            y_mid.append(y[i])

        else:
            x_sell.append(x[i])
            y_sell.append(y[i])

    return x_buy, x_mid, x_sell, y_buy, y_mid, y_sell


# 指定された時間間隔用のDBHelperを作成
def createResultXminHelper(dbPath, Xmin):

    if Xmin == 1:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_1MIN_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 5:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_5MIN_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 10:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_10MIN_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 30:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_30MIN_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 60:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_1HOUR_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 240:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_4HOUR_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    elif Xmin == 1440:
        resultXminHelper = DBHelper(dbPath, DBConstants.RESULT_1DAY_TBL, DBConstants.RESULT_COLUMNS)
        return resultXminHelper

    else:
        return None

# 指定された時間間隔用のDBHelperを作成
def createIndicatorXminHelper(dbPath, Xmin):

    if Xmin == 1:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_1MIN_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 5:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_5MIN_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 10:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_10MIN_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 30:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_30MIN_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 60:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_1HOUR_FORMAT_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 240:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_4HOUR_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    elif Xmin == 1440:
        indicatorXminHelper = DBHelper(dbPath, DBConstants.INDICATOR_1DAY_TBL, DBConstants.INDICATOR_COLUMNS)
        return indicatorXminHelper

    else:
        return None


if __name__ == '__main__':

    x,y = readData()

    for i in range(198,200):
        print (x[i])
        print (y[i])

    x_train, y_train, x_test, y_test = splitData(x,y,0.3)

    x_test_b, x_test_m, x_test_s, y_test_b, y_test_m, y_test_s = divideData(x_test, y_test)


    supervisor_labels_ph = tf.placeholder("float",[None,3])
    input_ph = tf.placeholder("float", [None,INDI_LAST-1])
    keep_prob = tf.placeholder("float")

    # 学習用と評価用のfeed_dictを用意
    feed_dict_train = {input_ph:x_train, supervisor_labels_ph:y_train}
    feed_dict_test = {input_ph:x_test, supervisor_labels_ph:y_test}

    feed_dict_test_b = {input_ph: x_test_b, supervisor_labels_ph: y_test_b}
    feed_dict_test_m = {input_ph: x_test_m, supervisor_labels_ph: y_test_m}
    feed_dict_test_s = {input_ph: x_test_s, supervisor_labels_ph: y_test_s}

#    feed_dict_train = {input_ph: x_train, supervisor_labels_ph: y_train, keep_prob: 1.0}
#    feed_dict_supervisor = {input_ph: x_test, supervisor_labels_ph: y_test, keep_prob: 1.0}

    with tf.Session() as sess:

        y_output = inference(input_ph)
        #y_output = inference(input_ph, keep_prob)

        loss = loss(y_output,supervisor_labels_ph)

        train_step = optimize(loss)

        acc = accuracy(y_output, supervisor_labels_ph)

        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(20001):
            sess.run(train_step, feed_dict=feed_dict_train)

            if step%100==0:
                print("train     :" + str(sess.run(loss, feed_dict=feed_dict_train)))
                print("supervisor:" + str(sess.run(loss, feed_dict=feed_dict_test)))

                train_accuracy = sess.run(acc, feed_dict=feed_dict_train)
                print("step %d, train accuracy %g" % (step, train_accuracy))

                test_accuracy = sess.run(acc, feed_dict=feed_dict_test)
                print("step %d, test  accuracy %g" % (step, test_accuracy))

                test_accuracy_b = sess.run(acc, feed_dict=feed_dict_test_b)
                print("step %d, test buy  accuracy %g" % (step, test_accuracy_b))

                test_accuracy_m = sess.run(acc, feed_dict=feed_dict_test_m)
                print("step %d, test mid  accuracy %g" % (step, test_accuracy_m))

                test_accuracy_s = sess.run(acc, feed_dict=feed_dict_test_s)
                print("step %d, test sell accuracy %g" % (step, test_accuracy_s))





