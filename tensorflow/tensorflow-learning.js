import * as tf from '@tensorflow/tfjs';

//定义线性回归模型
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// 定义损失函数和优化方法
model.compile({
    loss: 'meanSquaredError', 
    optimizer: 'sgd'
});

// 添加训练数据
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// 训练并预测
model.fit(xs, ys).then(() => {
    model.predict(tf.tensor2d([5], [1, 1])).print();
});