export function mse(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        sum += Math.pow(yTrue[i] - yPred[i], 2);
    }
    return sum / yTrue.length;
}

export function msePrime(yTrue, yPred) {
    let result = [];
    for (let i = 0; i < yTrue.length; i++) {
        result.push(2 * (yPred[i] - yTrue[i]) / yTrue.length);
    }
    return result;
}

export function binaryCrossEntropy(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        sum += -yTrue[i] * Math.log(yPred[i]) - (1 - yTrue[i]) * Math.log(1 - yPred[i]);
    }
    return sum / yTrue.length;
}

export function binaryCrossEntropyPrime(yTrue, yPred) {
    let result = [];
    for (let i = 0; i < yTrue.length; i++) {
        result.push(((1 - yTrue[i]) / (1 - yPred[i]) - yTrue[i] / yPred[i]) / yTrue.length);
    }
    return result;
}
