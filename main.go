package main

import (
	"encoding/json"
	"fmt"
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
		calcDiscriminantFunction,
	)

	err := myAi.ReadFromCSV("test_data.csv")
	if err != nil {
		panic(err)
	}

	myAi.TrainSingleEra()
	answerByte, err := json.MarshalIndent(myAi, "", "    ")
	if err != nil {
		panic(err)
	}

	fmt.Println(string(answerByte))
}

func calcDiscriminantFunction(numbersSet []float64) (float64, error) {
	const op string = "calcDiscriminantFunction"
	result := math.Pow(numbersSet[1], 2) - 4*numbersSet[0]*numbersSet[2]
	if math.IsNaN(result) {
		return 0, fmt.Errorf("%s: NaN", op)
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("%s: Infinity", op)
	}

	return result, nil
}

func sigmoidFunc(x float64) (float64, error) {
	const op string = "sigmoidFunc"
	result := 1 / (1 + math.Exp(-x))
	if math.IsNaN(result) {
		return 0, fmt.Errorf("%s: NaN", op)
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("%s: Infinity", op)
	}

	return result, nil
}

func antiSigmoidFunc(x float64) (float64, error) {
	const op string = "antiSigmoidFunc"
	result := (1 - x) * x
	if math.IsNaN(result) {
		return 0, fmt.Errorf("%s: NaN", op)
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("%s: Infinity", op)
	}

	return result, nil
}

func tangensoidFunc(x float64) (float64, error) {
	const op string = "tangensoidFunc"
	result := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	if math.IsNaN(result) {
		return 0, fmt.Errorf("%s: NaN", op)
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("%s: Infinity", op)
	}

	return result, nil
}

func antiTangensoidFunc(x float64) (float64, error) {
	const op string = "antiTangensoidFunc"
	result := 1 - x*x
	if math.IsNaN(result) {
		return 0, fmt.Errorf("%s: NaN", op)
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("%s: Infinity", op)
	}

	return result, nil
}
