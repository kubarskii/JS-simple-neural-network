import BaseLayer from "./baseLayer.js";
import { randn } from "../utils/rand.js";
import { Matrix, Vector, Tensor } from "../utils/math.js";

export default class DenseLayer extends BaseLayer {
    constructor(inputSize, outputSize) {
        super();
        this.params = { inputSize, outputSize }
        this.weights = new Matrix(
            ...Array.from({ length: outputSize }, () =>
                new Vector(...Array.from({ length: inputSize }, () => Math.random())))
        );
        this.bias = new Vector(...Array.from({ length: outputSize }, () => Math.random()));
    }

    forward(input) {
        if (!(input instanceof Vector)) {
            throw new Error("Input must be a Vector.");
        }
        this.input = input;
        const weightedInput = this.weights.data.map(row => row.dot(input));
        this.output = new Vector(...weightedInput).add(this.bias);
        return this.output;
    }

    backward(outputGradient, learningRate) {
        if (!(outputGradient instanceof Vector)) {
            throw new Error("Argument must be an instance of Vector");
        }
        const weightsGradient = new Matrix(...this.weights.data.map((row, i) =>
            new Vector(...row.data.map((_, j) => outputGradient.data[i] * this.input.data[j]))
        ));
        const inputGradient = new Vector(
            ...this.input.data.map((_, i) =>
                this.weights.data.reduce((acc, row, j) => acc + row.data[i] * outputGradient.data[j], 0)
            )
        );
        this.weights = new Matrix(...this.weights.data.map((row, i) =>
            new Vector(...row.data.map((val, j) => val - learningRate * weightsGradient.data[i].data[j]))
        ));
        this.bias = new Vector(...this.bias.data.map((val, i) => val - learningRate * outputGradient.data[i]));
        return inputGradient;
    }
}