import * as tf from "@tensorflow/tfjs";

// 1. Bayesian Regression (basic simulation)
function bayesianRegression(x, y, numSamples = 1000) {
  let samples = [];
  for (let i = 0; i < numSamples; i++) {
    let a = Math.random();
    let b = Math.random();
    let sigma = Math.random();
    let likelihood = y.reduce((acc, yi, idx) => {
      const pred = a + b * x[idx];
      return acc * Math.exp(-Math.pow(yi - pred, 2) / (2 * sigma));
    }, 1);

    samples.push({ a, b, sigma, likelihood });
  }
  samples.sort((a, b) => b.likelihood - a.likelihood);
  return samples[0];
}

// 2. Quantile Regression Forests (basic simulation)
class QuantileRegressionForest {
  constructor(numTrees = 100) {
    this.numTrees = numTrees;
    this.trees = [];
  }

  fit(X, y) {
    this.trees = Array.from({ length: this.numTrees }, () =>
      X.map((x, i) => ({ x, y: y[i] }))
    );
  }

  predict(X, quantile = 0.5) {
    return X.map((xi) => {
      const preds = this.trees.flatMap((tree) =>
        tree.filter((node) => node.x === xi).map((node) => node.y)
      );
      preds.sort((a, b) => a - b);
      return preds[Math.floor(quantile * preds.length)];
    });
  }
}

// 3. Neural Network Regression
async function neuralNetworkRegression() {
  const xs = tf.tensor1d([0.1, 0.2, 0.3, 0.4, 0.5]);
  const ys = tf.tensor1d([1.1, 1.9, 3.0, 3.8, 5.1]);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 10, inputShape: [1], activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: "adam", loss: "meanSquaredError" });
  await model.fit(xs, ys, { epochs: 100 });
  const pred = model.predict(tf.tensor1d([0.6]));
  pred.print();
}

// Run all models
const x = [1, 2, 3, 4, 5];
const y = [2.2, 4.0, 5.8, 8.1, 10.3];
console.log("Bayesian Regression:", bayesianRegression(x, y));
const qrf = new QuantileRegressionForest();
qrf.fit(x, y);
console.log("Quantile Regression Forests:", qrf.predict([3], 0.9));
neuralNetworkRegression();
