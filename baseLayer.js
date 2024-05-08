/**
 * @abstract
 */
export default class BaseLayer {
    constructor() {
        this.input = []
        this.output = []
    }

    /**
     * 
     * @param {Array<number>} input 
     */
    forward(input) {
        throw new Error("forward method not implemented")
    }

    /**
     * 
     * @param {Array<number>} outputGradient 
     * @param {number} learningRate 
     */
    backward(outputGradient, learningRate) {
        throw new Error("backward method not implemented")
    }
}