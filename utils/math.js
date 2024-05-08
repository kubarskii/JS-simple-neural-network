export class Vector extends Array {
    constructor(...elements) {
        if (elements.length === 1) {
            super(1);
            this[0] = elements[0];
        } else {
            super(...elements);
        }
    }

    // Element-wise addition of two vectors
    add(other) {
        // debugger;
        if (this.length !== other.length) {
            throw new Error("Both vectors must be of the same length to add.");
        }
        return new Vector(...this.map((v, i) => parseFloat(v) + parseFloat(other[i])));
    }

    // Element-wise subtraction of two vectors
    subtract(other) {
        if (this.length !== other.length) {
            throw new Error("Both vectors must be of the same length to subtract.");
        }
        return new Vector(...this.map((v, i) => v - other[i]));
    }

    // Dot product of two vectors
    mul(other) {
        if (typeof other === "number") {
            return new Vector(...this.map(v => v * other));
        }
        if (this.length !== other.length) {
            throw new Error("Both vectors must be of the same length to compute the dot product.");
        }
        return this.reduce((acc, curr, i) => acc + curr * other[i], 0);
    }
}

export class Matrix extends Vector {
    constructor(...rows) {
        super(...rows);
    }
    // Matrix addition
    add(other) {
        if (this.length !== other.length) {
            throw new Error("Both matrices must have the same dimensions to add.");
        }
        return new Matrix(...this.map((row, i) => row.add(other[i])));
    }

    // Matrix subtraction
    subtract(other) {
        if (this.length !== other.length) {
            throw new Error("Both matrices must have the same dimensions to subtract.");
        }
        return new Matrix(...this.map((row, i) => row.subtract(other[i])));
    }

    // Matrix multiplication
    mul(other) {
        if (typeof other === "number") {
            return new Matrix(...this.map(row => row.mul(other)));
        }
        if (other instanceof Matrix) {
            // debugger;
            if (this[0].length !== other.length) {
                throw new Error("The number of columns in the first matrix must equal the number of rows in the second matrix.");
            }
            return new Matrix(...this.map(row => {
                return new Vector(...Array.from({ length: other[0].length }, (_, colIndex) =>
                    row.mul(new Vector(...other.map(row => row[colIndex])))
                ))
            }
            ));
        }
        if (other instanceof Vector) {
            if (this[0].length !== other.length) {
                throw new Error("The number of columns in the matrix must equal the length of the vector.");
            }
            return new Vector(...this.map(row => row.mul(other)));
        }
        throw new Error("The object to multiply must be either a Vector or another Matrix.");
    }

    get T() {
        return this.transpose();
    }

    transpose() {
        return new Matrix(...Array.from({ length: this[0].length }, (_, colIndex) =>
            new Vector(...this.map(row => row[colIndex]))
        ));
    }
}

export const v = (...args) => new Vector(...args);
export const m = (...args) => new Matrix(...args); 