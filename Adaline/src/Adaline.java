import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class Adaline extends NeuralNetwork {
	
	public Adaline(int inputNeuronsCount, int outputNeuronsCount, double learnRate, TransferFunctionType transferFunction) {
		this.createNetwork(inputNeuronsCount, outputNeuronsCount, learnRate,transferFunction);
	}
	
	private void createNetwork(int inputNeuronsCount, int outputNeuronsCount, double learnRate,
			TransferFunctionType transferFunction) {
		this.setNetworkType(NeuralNetworkType.ADALINE);
		NeuronProperties inNeuronProperties= new NeuronProperties();
		inNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);
		Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inNeuronProperties);
		inputLayer.addNeuron(new BiasNeuron());
		
		this.addLayer(inputLayer);
		
		NeuronProperties outNeuronProperties = new NeuronProperties();
		if (transferFunction == TransferFunctionType.LINEAR) {
			outNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);
		} else {
			outNeuronProperties.setProperty("transferFunction", TransferFunctionType.RAMP);
			outNeuronProperties.setProperty("transferFunction.slope", new Double(1));
			outNeuronProperties.setProperty("transferFunction.yHigh", new Double(1));
			outNeuronProperties.setProperty("transferFunction.xHigh", new Double(1));
			outNeuronProperties.setProperty("transferFunction.yLow", new Double(-1));
			outNeuronProperties.setProperty("transferFunction.xLow", new Double(-1));
		}
		Layer outputLayer = LayerFactory.createLayer(outputNeuronsCount, outNeuronProperties);
		this.addLayer(outputLayer);
		
		ConnectionFactory.fullConnect(inputLayer, outputLayer);
		NeuralNetworkFactory.setDefaultIO(this);

		LMS l = new LMS();
		l.setLearningRate(learnRate);
		this.setLearningRule(l);
	}
}
