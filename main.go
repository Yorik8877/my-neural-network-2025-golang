package main

import (
	"math"
	"my-neural-network-2025-golang/ai"
)

func main() {
	hiddenWeights := [][]float64{
		{0.1, 0.9, 0.31},
		{0.4, 0.7, 0.11},
		{0.3, 0.2, 0.27},
	}

	outputWeights := []float64{0.47, 0.51, 0.67}

	myAi := ai.NewNeuralNetwork(
		hiddenWeights, outputWeights,
		sigmoidFunc, antiSigmoidFunc,
		calcDiscriminantFunction, 10000, 0.7, 0.3,
	)

	myAi.ReadFromCSV("test_data.csv")
	myAi.TrainAI()
}

func calcDiscriminantFunction(numbersSet []float64) float64 {
	return math.Pow(numbersSet[1], 2) - 4*numbersSet[0]*numbersSet[2]
}

func sigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
func antiSigmoidFunc(x float64) float64 {
	return (1 - x) * x
}

func tangensoidFunc(x float64) float64 {
	return (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
}
func antiTangensoidFunc(x float64) float64 {
	return 1 - x*x
}
