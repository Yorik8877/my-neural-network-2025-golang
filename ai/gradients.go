package ai

func (ai NeuralNetwork) calcGradients(hiddenGradients [][]float64, outputGradients []float64) {
	for gradIdx, grad := range outputGradients {
		ai.PreviousOutputWeightDeltas[gradIdx] = ai.calcDelta(
			grad,
			ai.PreviousOutputWeightDeltas[gradIdx],
		)
	}

	for gradsIdx, gradients := range hiddenGradients {
		for gradIdx, grad := range gradients {
			ai.PreviousHiddenWeightDeltas[gradsIdx][gradIdx] = ai.calcDelta(
				grad,
				ai.PreviousHiddenWeightDeltas[gradsIdx][gradIdx],
			)
		}
	}
}
