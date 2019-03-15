//实现knn L1距离度量
export function knn(train_data,train_label){
    const train_x = tf.tensor2d(train_data);
    const train_y = tf.tensor1d(train_label,'int32');

    return function(x){
        var res = [];
        x.map((point)=>{
            const input_tensor = tf.tensor1d(point);
            const distance = tf.sum(tf.abs(tf.sub(input_tensor,train_x)),1);
            const index = tf.argMin(distance, 0);
            res.push(train_label[index.dataSync()[0]]);
        });
        return res;
    }
}