'use strict';

var _tfjs = require('@tensorflow/tfjs');

var tf = _interopRequireWildcard(_tfjs);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

// //定义线性回归模型
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// // 定义损失函数和优化方法
// model.compile({
//     loss: 'meanSquaredError', 
//     optimizer: 'sgd'
// });

// // 添加训练数据
// const xs = tf.tensor([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor([2, 4, 6, 8], [4, 1]);

// // 训练并预测
// model.fit(xs, ys).then(() => {
//     model.predict(tf.tensor([5], [1, 1])).print();
// });

//add,sub,mul只能在同规模的张量之间和标量进行
//matMul 矩阵乘法
// const b = tf.tensor([[1,2,3],[4,5,6]]);
// b.print();
// const a= tf.zeros([3,3]);
// a.print();
// const c = tf.ones([2,3]);
// c.print();
// let v = tf.variable(a);
// v.print();
// v = c;
// v.print();
// const d = b.square().print();
// const e = tf.mul(b,c);
// e.print();
// const f = tf.scalar(3);
// tf.sub(b,f).print();
// tf.matMul(b,a).print();
// b.transpose().print();
// //const g = tf.oneHot(tf.tensor1d([0.0,1.0,2.0]),3).print(); 
// b.min(0).print();
// b.mean(0).print();
// b.sum(0).print();
//b.dispose();
// tf.tidy(()=>{
//     return neededData;
// });

//实现线性回归模型 y = w * x + b
var tx = [1, 2, 3, 4, 5];
var ty = [2, 4, 6, 8, 10];
var w = tf.variable(tf.scalar(Math.random()));
var b = tf.variable(tf.scalar(Math.random()));
var train_x = (0, _tfjs.tensor1d)(tx);
var train_y = (0, _tfjs.tensor1d)(ty);

//定义学习率和迭代次数
var iterations = 200;
var learningRate = 0.5;

var f = function f(x) {
    return w.mul(x).add(b);
};

//选择优化器sgd,momentum,adam等
var optimizer = tf.train.adam(learningRate);

//定义均方差损失函数
var loss = function loss(pred, label) {
    return pred.sub(label).square().mean();
};

//开始训练
for (var iter = 0; iter < iterations; iter++) {
    optimizer.minimize(function () {
        var loss_var = loss(f(train_x), train_y);
        console.log('loss: ');
        loss_var.print();
        console.log('w: ');
        w.print();
        //b.print();
        return loss_var;
    });
}
