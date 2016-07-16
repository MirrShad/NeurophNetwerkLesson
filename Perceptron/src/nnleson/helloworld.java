package nnleson;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;

public class helloworld {

	public static void main(String[] args) {
		//create trainingSet(logical AND function)
		DataSet trainingSet = new DataSet(2,1);
		trainingSet.addRow(new DataSetRow(new double[]{0,0},new double[]{0}));
		trainingSet.addRow(new DataSetRow(new double[]{0,1},new double[]{0}));
		trainingSet.addRow(new DataSetRow(new double[]{1,0},new double[]{0}));
		trainingSet.addRow(new DataSetRow(new double[]{1,1},new double[]{1}));
		//create perceptron neural network
		NeuralNetwork myPerceptron = new Perceptron(2,1);
		
		//learn the training set
		myPerceptron.learn(trainingSet);
		//test perceptron
		System.out.println("testing trained perceptron");
		testNeuralNetwork(myPerceptron, trainingSet);
		//save trained perceptron
		myPerceptron.save("mySamplePerceptron.nnet");
		//load saved neural network
		NeuralNetwork loadedPerceptron=NeuralNetwork.createFromFile("mySamplePerceptron.nnet");
		
		//test loaded neural network
		System.out.println("Testing loaded perceptron");
		testNeuralNetwork(loadedPerceptron,trainingSet);
	}

	public static void testNeuralNetwork(NeuralNetwork nnet,DataSet tset)
	{
		for(DataSetRow dataRow:tset.getRows())
		{
			nnet.setInput(dataRow.getInput());
			nnet.calculate();
			double[] networkOutput=nnet.getOutput();
			System.out.print("Input:"+Arrays.toString(dataRow.getInput()));
			System.out.println("Output:"+Arrays.toString(networkOutput));
		}
	}
}
