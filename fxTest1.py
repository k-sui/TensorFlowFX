import tensorflow as tf
import csv
import random

#csvファイル内の列の位置を示す定数
TIME = 0
START_PRICE = 1
MIN_PRICE = 2
MAX_PRICE = 3
END_PRICE = 4
INDI_FIRST = 5  # インジケータが入る最初の列
INDI_LAST = 14  # インジケータが入る最後の列

# 金額の上下を決めるための閾値を割合で指定
DIF_RATA_THRESHOLD = 0.0003

# 隠れ層の数
HIDDEN_LAYER_NUM_1 = 50
HIDDEN_LAYER_NUM_2 = 25



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


    #DropOut
#    dropout = tf.nn.dropout(hidden2, keep_prob)

    with tf.name_scope('softmax'):
        weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NUM_2, 3], stddev=1.0), name='weights')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        y = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)
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


def readData(filePass):

    # csvから読み込んだデータを保持する配列
    x_data = []

    # 正解データを保持する配列
    y_ans = []

    # TODO CSVファイルからデータの読み込み
    rf = open(filePass,'r')
    rawData = csv.reader(rf)

    # 価格上下データの分布の確認用
    up = 0
    mid = 0
    down = 0

    prevRow = None
    thisRow = None

    # ラベルを作成。前の時間の金額よりも○%の上下、それよりも変化が小さい場合の3つに分ける [up, mid, down]
    for thisRow in rawData:

        # 最左列の時間以外を保存
        x_data.append([float(thisRow[i]) for i in range(1,INDI_LAST)])

        # 最初のデータの場合はスルー
        if prevRow is None:
            prevRow = thisRow
            continue

        # 閾値よりも上昇している場合
        if float(prevRow[END_PRICE]) * (1+DIF_RATA_THRESHOLD) < float(thisRow[END_PRICE]):
            y_ans.append([1., 0., 0.])
            up += 1
        # 閾値よりも下落している場合
        elif float(prevRow[END_PRICE]) * (1-DIF_RATA_THRESHOLD) > float(thisRow[END_PRICE]):
            y_ans.append([0., 0., 1.])
            down += 1
        # その他(=変化なし)の場合
        else:
            y_ans.append([0., 1., 0.])
            mid += 1

        prevRow = thisRow

    # 価格上下データの分布の表示
    print("up:"+ str(up*100/(up+mid+down)) + "%")
    print("mid:"+ str(mid*100/(up+mid+down)) + "%")
    print("down:"+ str(down*100/(up+mid+down)) + "%")

    return x_data, y_ans

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
        rnd = random.random()
        if rnd >= testRatio:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    x,y = readData("/home/k-sui/fxdata/1h_2000-2015_indicator.csv")

    for i in range(1,200):
        print (x[i])
        print (y[i])

    x_train, y_train, x_test, y_test = splitData(x,y,0.3)


    supervisor_labels_ph = tf.placeholder("float",[None,3])
    input_ph = tf.placeholder("float", [None,INDI_LAST-1])
    keep_prob = tf.placeholder("float")

    # 学習用と評価用のfeed_dictを用意
    feed_dict_train = {input_ph:x_train, supervisor_labels_ph:y_train}
    feed_dict_supervisor = {input_ph:x_test, supervisor_labels_ph:y_test}

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

        for step in range(1000):
            sess.run(train_step, feed_dict=feed_dict_train)

            if step%100==0:
                print("train     :" + str(sess.run(loss, feed_dict=feed_dict_train)))
                print("supervisor:" + str(sess.run(loss, feed_dict=feed_dict_supervisor)))

                train_accuracy = sess.run(acc, feed_dict=feed_dict_supervisor)
                print("step %d, training accuracy %g" % (step, train_accuracy))


