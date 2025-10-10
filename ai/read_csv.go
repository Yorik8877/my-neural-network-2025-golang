package ai

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
)

func (ai *NeuralNetwork) ReadFromCSV(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '\t'

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			return err
		}

		var buffer []float64
		for _, value := range record {
			parsedValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(err)
			}

			buffer = append(buffer, parsedValue)
		}

		ai.dataSet = append(ai.dataSet, buffer)
	}

	return nil
}
