
import { NativeModules, Image } from 'react-native';
import uuid from 'uuid/v1';

const { RNTensorFlowInference, RNTensorFlowGraph, RNTensorFlowGraphOperations } = NativeModules;

class TensorFlowOperation {
  constructor(id, opName) {
    this.id = id
    this.opName = opName
  }

  inputListLength(name) {
    return RNTensorFlowGraphOperations.inputListLength(this.id, this.opName, name);
  }

  name() {
    return RNTensorFlowGraphOperations.name(this.id, this.opName)
  }

  numOutputs() {
    return RNTensorFlowGraphOperations.numOutputs(this.id, this.opName)
  }

  output(index) {
    return RNTensorFlowGraphOperations.output(this.id, this.opName, index)
  }

  outputList(index, length) {
    return RNTensorFlowGraphOperations.outputList(this.id, this.opName, index, length)
  }

  outputListLength(name) {
    return RNTensorFlowGraphOperations.outputListLength(this.id, this.opName, name)
  }

  type() {
    return RNTensorFlowGraphOperations.type(this.id, this.opName)
  }
}

class TensorFlowGraph {
  constructor(id) {
    this.id = id
  }

  importGraphDef(graphDef) {
    return RNTensorFlowGraph.importGraphDef(this.id, graphDef)
  }

  importGraphDefWithPrefix(graphDef, prefix) {
    return RNTensorFlowGraph.importGraphDefWithPrefix(this.id, graphDef, prefix)
  }

  toGraphDef() {
    return RNTensorFlowGraph.toGraphDef(this.id)
  }

  operation(name) {
    return new TensorFlowOperation(this.id, name)
  }

  close() {
    return RNTensorFlowGraph.close(this.id)
  }
}

class TensorFlowInference {

  constructor(modelLocation) {
    this.id = uuid()
    const resolvedModelLocation = Image.resolveAssetSource(modelLocation) != null
      ? Image.resolveAssetSource(modelLocation).uri
      : modelLocation
    this.init = RNTensorFlowInference.initTensorFlowInference(this.id, resolvedModelLocation)
    this.tfGraph = new TensorFlowGraph(this.id)
  }

  async feed(data) {
    await this.init
    return RNTensorFlowInference.feed(this.id, data)
  }

  async feedImage(data) {
    await this.init
    return RNTensorFlowInference.feedImage(this.id, data)
  }

  async run(outputNames, withStats) {
    await this.init
    return RNTensorFlowInference.run(this.id, outputNames, withStats !== undefined ? withStats : false)
  }

  async fetch(outputName) {
    await this.init
    return RNTensorFlowInference.fetch(this.id, outputName)
  }

  async graph() {
    await this.init
    return this.tfGraph
  }

  async close() {
    await this.init
    return RNTensorFlowInference.close(this.id)
  }
}

export default TensorFlowInference;
