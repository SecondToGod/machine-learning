export function nn_model(){
    const model = tf.sequential();//序列化网络模型
    model.add(tf.layers.dense({ //dense表全连接层
        units: 32 , inputShape: [784] //32个神经单元
    }));
    model.add(tf.layers.dense({
        units: 256 , //256个神经单元
    }));
    model.add(tf.layers.dense({
        units: 10 , kernelInitializer: 'varianceScaling', activation: 'softmax'
    }));
    return model;
}

//网络初始化
const model = new nn_model();
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);
model.compile({
    optimizer : optimizer,
    loss : 'categoricalCrossentropy',
    metrics: ['accuracy']
});

//训练过程
async function train(){
    const batch_size = 16; //批大小
    const train_batches = 100; //训练数据批数

    const test_batch_size = 100; //测试批大小
    const test_iteration_frequency = 5; //测试频率

    for(let i=0;i<train_batches;i++){
        const batch = data.get(batch_size);// 获取批训练数据

        let testBatch,validationData;
        if(i % test_iteration_frequency === 0 && i>0){
            testBatch = testData.get(test_batch_size);// 获取测试数据
            validationData = [
                testBatch.features.reshape([test_batch_size,784]),testBatch.labels
            ];
        }
        
        const history = await model.fit(
            batch.features.reshape([batch_size,784]),
            batch.labels,
            {
                batch_size: batch_size,
                validationData,
                epochs: 1
            }
        );
        batch.features.dispose();
        batch.labels.dispose();
        if(testBatch !== null){
            testBatch.features.dispose();
            testBatch.labels.dispose();
        }
        await tf.nextFrame();
    }
}