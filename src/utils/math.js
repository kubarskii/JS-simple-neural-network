export class Tensor {
    constructor(data) {
        this.data = data;
    }

    map(func) {
        // Generic map function, need to be specific in child classes
        throw new Error("Map function not implemented for base Tensor class.");
    }

    add(tensor) {
        if (!(tensor instanceof Tensor)) {
            throw new Error("Argument must be an instance of Tensor.");
        }
        if (this.shape().toString() !== tensor.shape().toString()) {
            throw new Error("Both tensors must have the same shape to add.");
        }
        return this._elementWiseOperation(tensor, (a, b) => a + b);
    }

    subtract(tensor) {
        if (!(tensor instanceof Tensor)) {
            throw new Error("Argument must be an instance of Tensor.");
        }
        if (this.shape().toString() !== tensor.shape().toString()) {
            throw new Error("Both tensors must have the same shape to subtract.");
        }
        return this._elementWiseOperation(tensor, (a, b) => a - b);
    }

    mul(tensor) {
        if (typeof tensor === 'number') {
            return this._elementWiseOperation(new Vector(this.data.map(v => v * tensor)), (a, b) => a * b);
        } else if (tensor instanceof Tensor) {
            return this._elementWiseOperation(tensor, (a, b) => a * b);
        } else {
            throw new Error("Argument must be a number or an instance of Tensor.");
        }
    }

    _elementWiseOperation(tensor, operation) {
        const result = this.data.map((val, i) => {
            if (Array.isArray(val)) {
                return val.map((subval, j) => operation(subval, tensor.data[i][j]));
            } else {
                return operation(val, tensor.data[i]);
            }
        });
        return new this.constructor(result);
    }

    shape() {
        return Array.isArray(this.data[0]) ? [this.data.length, this.data[0].length] : [this.data.length];
    }

    toString() {
        return JSON.stringify(this.data);
    }

    transpose() {
        throw new Error("Transpose method should be implemented in subclasses.");
    }

    get T() {
        return this.transpose();
    }
}

export class Vector extends Tensor {
    constructor(...elements) {
        super(elements);
    }

    dot(other) {
        if (!(other instanceof Vector)) {
            throw new Error("Argument must be an instance of Vector.");
        }
        if (this.data.length !== other.data.length) {
            throw new Error("Both vectors must have the same length to compute the dot product.");
        }
        return this.data.reduce((acc, curr, idx) => acc + curr * other.data[idx], 0);
    }

    add(vector) {
        if (!(vector instanceof Vector)) {
            throw new Error("Argument must be an instance of Vector.");
        }
        if (this.data.length !== vector.data.length) {
            throw new Error("Both vectors must have the same length to add.");
        }
        return new Vector(...this.data.map((val, idx) => val + vector.data[idx]));
    }

    scalarMul(scalar) {
        return new Vector(...this.data.map(value => value * scalar));
    }

    map(func) {
        return new Vector(...this.data.map(func));
    }

    transpose() {
        return new Matrix(...this.data.map(val => new Vector(val)));
    }
}

export class Matrix extends Tensor {
    constructor(...rows) {
        if (rows.some(row => !(row instanceof Vector))) {
            throw new Error("All rows must be instances of Vector.");
        }
        super(rows);
    }

    map(func) {
        return new Matrix(...this.data.map(row => row.map(func)));
    }

    mul(other) {
        if (typeof other === "number") {
            return new Matrix(...this.data.map(row => row.scalarMul(other)));
        } else if (other instanceof Vector) {
            // Check dimensions for matrix-vector multiplication
            if (this.data[0].data.length !== other.data.length) {
                throw new Error("The number of columns in the matrix must equal the length of the vector.");
            }
            // Perform matrix-vector multiplication correctly using the Vector's data
            return new Vector(...this.data.map(row =>
                row.data.reduce((acc, value, idx) => acc + value * other.data[idx], 0)
            ));
        } else if (other instanceof Matrix) {
            if (this.data[0].data.length !== other.data.length) {
                throw new Error("The number of columns in the first matrix must equal the number of rows in the second matrix.");
            }
            let transposedOther = other.transpose();
            return new Matrix(...this.data.map(row =>
                new Vector(...transposedOther.data.map(col =>
                    row.data.reduce((sum, value, idx) => sum + value * col.data[idx], 0)
                ))
            ));
        } else {
            throw new Error("Argument must be either a number, Vector, or another Matrix.");
        }
    }

    transpose() {
        const transposedData = this.data[0].data.map((_, colIndex) => this.data.map(row => row.data[colIndex]));
        return new Matrix(...transposedData.map(col => new Vector(...col)));
    }

    _elementWiseOperation(tensor, operation) {
        const result = this.data.map((row, i) => {
            if (row instanceof Vector) {
                return new Vector(...row.data.map((val, j) => operation(val, tensor.data[i].data[j])));
            }
            throw new Error("Row must be an instance of Vector.");
        });
        return new Matrix(...result);
    }

    add(tensor) {
        if (!(tensor instanceof Matrix)) {
            throw new Error("Argument must be an instance of Matrix.");
        }
        if (this.shape().toString() !== tensor.shape().toString()) {
            throw new Error("Both matrices must have the same shape to add.");
        }
        return this._elementWiseOperation(tensor, (a, b) => a + b);
    }
}
