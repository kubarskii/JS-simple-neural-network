import BaseLayer from "./baseLayer.js";
import { randn } from "./utils/rand.js";

export default class DenseLayer extends BaseLayer {
    /**
     * 
     * @param {number} inputSize 
     * @param {number} outputSize 
     */
    constructor(inputSize, outputSize) {
        super();
        this.weights = randn(outputSize, inputSize)
        this.bias = randn(outputSize, 1)
    }

    /**
     * 
     * @param {import('./utils/math.js').Matrix} input 
     * @returns {import('./utils/math.js').Matrix}
     */
    forward(input) {
        this.input = input
        return this.weights.mul(input).add(this.bias)
    }

    /**
     * 
     * @param {import('./utils/math.js').Matrix} outputGradient 
     * @param {number} learningRate 
     * @returns {import('./utils/math.js').Matrix}
     */
    backward(outputGradient, learningRate) {
        const weightsGradient = outputGradient.mul(this.input.T);
        const inputGradient = this.weights.T.mul(outputGradient);
        this.weights.subtract(weightsGradient.mul(learningRate))
        this.bias.subtract(outputGradient.mul(learningRate))
        return inputGradient
    }
}