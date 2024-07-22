import { getDataset, getClasses, getCrossValidationSets, getClassesAsNumber, getNumbers } from 'ml-dataset-iris';
import NN from './src/core/NN.js';
import DenseLayer from './src/layers/denseLayer.js';
import { mse, msePrime } from './src/utils/loss.js'
import { Matrix, Vector } from './src/utils/math.js';

const indexOfLargestValue = (arr) => arr.reduce((maxIndex, currentValue, currentIndex, array) => currentValue > array[maxIndex] ? currentIndex : maxIndex, 0)


function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Derivative of the sigmoid function
function sigmoidDerivative(x) {
    const sig = sigmoid(x);
    return sig * (1 - sig);
}

const sigmoidActivation = {
    func: sigmoid,
    derivative: sigmoidDerivative
};

async function trainOnIrisDataset() {
    try {
        const [irisDataset] = [getDataset(), getClassesAsNumber(), getCrossValidationSets(5), getClasses()];
        const classesMap = {
            setosa: 0,
            versicolor: 1,
            virginica: 2,
        }
        const classesArr = ['setosa', 'versicolor', 'virginica']
        const inputsArr = [];
        const outputsArr = []
        const irisDatasetCopy = irisDataset.slice(); // Copy the dataset
        const trainingSize = 90; // Number of samples in training data
        const shuffledIndices = [...Array(irisDatasetCopy.length).keys()].sort(() => Math.random() - 0.5);
        for (let i = 0; i < trainingSize; i++) {
            const index = shuffledIndices[i];
            const arrIndex = classesMap[irisDatasetCopy[index].at(-1)];
            const out = new Array(3).fill(0);
            out[arrIndex] = 1;
            outputsArr.push(new Vector(...out));
            inputsArr.push(new Vector(...irisDatasetCopy[index].slice(0, -1)));
        }
        const inputs = new Matrix(...inputsArr);
        const outputs = new Matrix(...outputsArr);
        const layers = [
            new DenseLayer(4, 10),
            new DenseLayer(10, 3),
        ]
        const activations = [sigmoidActivation, sigmoidActivation];
        const h = 0.00001;
        const mseErrorFunction = {
            func: mse,
            derivative: (yTrue, yPred) => {
                const result = [];
                for (let i = 0; i < yTrue.length; i++) {
                    const yPredH = [...yPred];
                    yPredH[i] += h;
                    result.push((mse(yTrue, yPredH) - mse(yTrue, yPred)) / h);
                }
                return result;
            }
        };
        const nn = new NN(layers, activations, mseErrorFunction);
        nn.train(inputs, outputs, 0.1, 100000);
        const rand = Math.floor(Math.random() * 150)
        const rand1 = Math.floor(Math.random() * 150)
        const rand2 = Math.floor(Math.random() * 150)
        const testInput = new Vector(...irisDataset[rand].slice(0, -1));
        const testInput1 = new Vector(...irisDataset[rand1].slice(0, -1));
        const testInput2 = new Vector(...irisDataset[rand2].slice(0, -1));
        const prediction = nn.predict(testInput);
        const prediction1 = nn.predict(testInput1);
        const prediction2 = nn.predict(testInput2);
        const p = indexOfLargestValue(prediction.data)
        const p1 = indexOfLargestValue(prediction1.data)
        const p2 = indexOfLargestValue(prediction2.data)
        console.log('Prediction:', classesArr[p], 'Should: ', irisDataset[rand].at(-1));
        console.log('Prediction1:', classesArr[p1], 'Should: ', irisDataset[rand1].at(-1));
        console.log('Prediction2:', classesArr[p2], 'Should: ', irisDataset[rand2].at(-1));
    } catch (error) {
        console.error('Error training on Iris dataset:', error);
    }
}

trainOnIrisDataset();
