import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class FaceService {
  late Interpreter _interpreter;
  late int inputSize;
  late List<String> _names;
  late List<List<double>> _embeddings;

  static const double cosineThreshold = 0.4;
  static const int embeddingSize = 512;

  FaceService();

  /// Load model + embeddings
  Future<void> loadModel() async {
    try {
      var options = InterpreterOptions()
        ..threads = 4; // multi-thread
      _interpreter = await Interpreter.fromAsset(
        'assets/model/facenet.tflite',
        options: options,
      );
    } catch (e) {
      _interpreter = await Interpreter.fromAsset('assets/model/facenet.tflite');
    }

    var inputShape = _interpreter.getInputTensor(0).shape;
    inputSize = inputShape[1]; // biasanya 160
    await _loadEmbeddings();
  }

  Future<void> _loadEmbeddings() async {
    final data =
        await rootBundle.loadString('assets/model/face_embeddings3.json');
    final Map<String, dynamic> jsonData = json.decode(data);

    _names = [];
    _embeddings = [];
    jsonData.forEach((name, embList) {
      for (var emb in embList) {
        if (emb.length == embeddingSize) {
          _names.add(name);
          _embeddings.add(List<double>.from(emb));
        }
      }
    });
  }

  /// Ambil crop wajah, resize ke inputSize, normalize [-1,1], lalu infer ke model
  List<double> getEmbedding(img.Image faceCrop) {
    final resized =
        img.copyResize(faceCrop, width: inputSize, height: inputSize);

    // normalisasi pixel -> [-1, 1]
    Float32List input = Float32List(inputSize * inputSize * 3);
    int index = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[index++] = (img.getRed(pixel) / 127.5) - 1.0;
        input[index++] = (img.getGreen(pixel) / 127.5) - 1.0;
        input[index++] = (img.getBlue(pixel) / 127.5) - 1.0;
      }
    }

    var inputTensor = input.reshape([1, inputSize, inputSize, 3]);
    var output = List.generate(1, (_) => List.filled(embeddingSize, 0.0));

    _interpreter.run(inputTensor, output);

    return output[0];
  }

  /// Bandingkan embedding dengan database
  String recognize(List<double> embedding) {
    double minDist = double.infinity;
    String identity = "Unknown";

    for (int i = 0; i < _embeddings.length; i++) {
      double dist = _cosineDistance(embedding, _embeddings[i]);
      if (dist < minDist) {
        minDist = dist;
        if (dist < cosineThreshold) {
          identity = _names[i];
        }
      }
    }
    return identity;
  }

  double _cosineDistance(List<double> a, List<double> b) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return 1 - (dot / (sqrt(normA) * sqrt(normB)));
  }
}
