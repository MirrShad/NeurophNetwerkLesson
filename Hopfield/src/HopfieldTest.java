import java.util.Arrays;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;
import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.nnet.learning.HopfieldLearning;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class HopfieldTest {
	public static double[] format(double[] data){
		for(int i=0;i<data.length;i++){
			if(data[i]==0)data[i]=-1;
		}
		return data;
	}
	
	public static void main(String args[]) {
		
		NeuronProperties neuronProperties = new NeuronProperties();
		neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
		neuronProperties.setProperty("bias", new Double(0.0D));
		neuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
		neuronProperties.setProperty("transferFunction.yHigh", new Double(1.0D));
		neuronProperties.setProperty("transferFunction.yLow", new Double(-1.0D));

		// create training set (H and T letter in 3x3 grid)
		DataSet trainingSet = new DataSet(4);
		trainingSet.addRow(new DataSetRow(new double[] { 
				1,1,-1,-1})); 
		
		trainingSet.addRow(new DataSetRow(new double[] { 
				1,-1,1,-1})); 
		
		// create hopfield network
		Hopfield myHopfield = new Hopfield(4, neuronProperties);
		myHopfield.setLearningRule(new StandHopfieldLearning());
		// learn the training set
		myHopfield.learn(trainingSet);

		// test hopfield network
		System.out.println("Testing network");

		DataSetRow h = new DataSetRow(new double[] { 
				1,1,-1,-1});
		trainingSet.addRow(h); 


		myHopfield.setInput(h.getInput());

		double[] networkOutput = null;
		double[] preNetworkOutput = null;
		while (true) {
			myHopfield.calculate();
			networkOutput = myHopfield.getOutput();
			if (preNetworkOutput == null) {
				preNetworkOutput = networkOutput;
				continue;
			}
			if (Arrays.equals(networkOutput, preNetworkOutput)) {
				break;
			}
			preNetworkOutput = networkOutput;
		}

		System.out.print("Input: " + Arrays.toString(h.getInput()));
		System.out.println(" Output: " + Arrays.toString(format(networkOutput)));

	}

}
