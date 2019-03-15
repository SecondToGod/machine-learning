//实现logistic回归
export function logistic_regression(train_data,train_label){
    const iterations = 200;
    const learningRate = 0.1;
    const optimizer = tf.train.adam(learningRate);

    //计算类别数量
    const num_labels = Array.from(new Set(train_label)).length;
    const num_data = train_label.length;

    const w = tf.variable(tf.zeros([2,num_labels]));
    const b = tf.variable(tf.zeros([num_labels]));

    const train_x = tf.tensor2d(train_data);
    const train_y = tf.tensor1d(train_label,'int32');

    function predict(x){
        return tf.softmax(tf.add(tf.matMul(x,w),b));
    }
    //定义交叉熵损失函数
    function loss(predictions,labels){
        const y = tf.oneHot(labels,num_labels);
        const entropy = tf.mean(tf.sub(tf.scalar(1),tf.sum(tf.mul(y,tf.log(predictions)),1)));
        return entropy;
    }
    //开始训练
    for(let iter=0;iter<iterations;iter++){
        optimizer.minimize(()=>{
            const loss_var = loss(predict(train_x),train_y);
            loss_var.print();
            return loss_var;
        });
    }
    return {
        predict: predict
    };
}