package nnleson;

import java.util.Arrays;
import java.util.Scanner;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class QuadrantJudge extends NeuralNetwork{

	public QuadrantJudge(int inputNeuronsCount) {
        this.createNetwork(inputNeuronsCount);
    }
	
	private void createNetwork(int inputNeuronsCount) {
        // �����������Ϊ ��֪��
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // ������Ԫ���� ����ʾ����Ĵ̼�
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        // ��������Ԫ���ɵ������
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        // �����������BiasNeuron����ʾ��Ԫƫ��
        inputLayer.addNeuron(new BiasNeuron());
        
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        Layer outputLayer = LayerFactory.createLayer(2, outputNeuronProperties);
        this.addLayer(outputLayer);
        
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        Neuron n=outputLayer.getNeuronAt(0);
        
        n.getInputConnections()[0].getWeight().setValue(0);
        n.getInputConnections()[1].getWeight().setValue(1);
        n.getInputConnections()[2].getWeight().setValue(0);
        
        n=outputLayer.getNeuronAt(1);
        n.getInputConnections()[0].getWeight().setValue(1);
        n.getInputConnections()[1].getWeight().setValue(0);
        n.getInputConnections()[2].getWeight().setValue(0);
        
    }
	public static void main(String[] args) {
		DataSet trainingSet = new DataSet(2, 2);
        QuadrantJudge perceptron = new QuadrantJudge(2);
        
        while(true)
        {
        	Scanner s = new Scanner(System.in);
        	double[] temp = new double[2];
        	System.out.println("����������꣺");
        	temp[0]=s.nextDouble();
        	System.out.println("�����������꣺");
        	temp[1]=s.nextDouble();
        	trainingSet.addRow(new DataSetRow(temp,new double[]{Double.NaN,Double.NaN}));
        	perceptron.setInput(trainingSet.getRowAt(0).getInput());
        	perceptron.calculate();
        	double[] networkOutput = perceptron.getOutput();
        	if(networkOutput[0]==0)
    		{
    			if(networkOutput[1]==0)
    			{
    				System.out.println(Arrays.toString(trainingSet.getRowAt(0).getInput())+"�ǵ�������");
    			}
    			else
    			{
    				System.out.println(Arrays.toString(trainingSet.getRowAt(0).getInput())+"�ǵ�������");
    			}
    		}
    		else
    		{
    			if(networkOutput[1]==0)
    			{
    				System.out.println(Arrays.toString(trainingSet.getRowAt(0).getInput())+"�ǵڶ�����");
    			}
    			else
    			{
    				System.out.println(Arrays.toString(trainingSet.getRowAt(0).getInput())+"�ǵ�һ����");
    			}
    		}
        	trainingSet.clear();
        }
	}

}
