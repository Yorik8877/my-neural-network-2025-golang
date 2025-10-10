package ai

import "math"

func (ai *NeuralNetwork) calcMSE(ideal float64) float64 {
	var mse float64
	for _, answer := range ai.Answers {
		mse += math.Pow(ideal-answer, 2)
	}

	return mse / float64(len(ai.Answers))
}
