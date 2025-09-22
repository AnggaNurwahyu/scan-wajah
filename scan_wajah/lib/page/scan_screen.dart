import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AbsensiScreen extends StatefulWidget {
  const AbsensiScreen({super.key});

  @override
  State<AbsensiScreen> createState() => _AbsensiScreenState();
}

class _AbsensiScreenState extends State<AbsensiScreen> {
  File? _image;
  bool _isProcessing = false;
  bool _showResult = false;
  String? _statusAbsensi;

  final String serverUrl = "http://192.168.110.62:5000/recognize";

  Future<void> _processImage(File file) async {
    setState(() {
      _isProcessing = true;
      _showResult = false;
    });

    var request = http.MultipartRequest("POST", Uri.parse(serverUrl));
    request.files.add(await http.MultipartFile.fromPath("file", file.path));

    var response = await request.send();

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      final data = json.decode(respStr);

      setState(() {
        _statusAbsensi = data["result"] != "Unknown"
            ? "✅ Absensi berhasil untuk ${data["result"]}"
            : "❌ Wajah tidak dikenali";
        _isProcessing = false;
        _showResult = true;
      });
    } else {
      setState(() {
        _statusAbsensi = "⚠️ Gagal menghubungi server";
        _isProcessing = false;
        _showResult = true;
      });
    }
  }

  Future<void> _getImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      final file = File(pickedFile.path);
      setState(() {
        _image = file;
      });
      await _processImage(file);
    }
  }

  void _showImageSourceOptions() {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF2A2A2A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext context) {
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt, color: Colors.white),
                title: const Text('Ambil dari Kamera',
                    style: TextStyle(color: Colors.white)),
                onTap: () {
                  Navigator.pop(context);
                  _getImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo, color: Colors.white),
                title: const Text('Pilih dari Galeri',
                    style: TextStyle(color: Colors.white)),
                onTap: () {
                  Navigator.pop(context);
                  _getImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1F1F1F),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Bar atas
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 15),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    InkWell(
                      onTap: () {},
                      child: Icon(
                        Icons.arrow_back_ios,
                        color: Colors.white.withAlpha((0.5 * 255).toInt()),
                      ),
                    ),
                    const Text(
                      "Absensi FaceNet (Server)",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(width: 24),
                  ],
                ),
              ),
              const SizedBox(height: 40),

              // Kotak ambil foto
              Center(
                child: GestureDetector(
                  onTap: _isProcessing ? null : _showImageSourceOptions,
                  child: Container(
                    width: 250,
                    height: 250,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: Colors.white, width: 1.5),
                    ),
                    child: Center(
                      child: _image != null
                          ? ClipRRect(
                              borderRadius: BorderRadius.circular(25),
                              child: Image.file(_image!,
                                  fit: BoxFit.cover,
                                  width: 250,
                                  height: 250),
                            )
                          : const Icon(Icons.camera_alt,
                              size: 50, color: Colors.white),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),

              const Text(
                '📋 Status Absensi:',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 10),

              if (_isProcessing) ...[
                const Center(
                    child: CircularProgressIndicator(color: Colors.greenAccent)),
                const SizedBox(height: 20),
              ],

              AnimatedOpacity(
                opacity: _showResult ? 1.0 : 0.0,
                duration: const Duration(milliseconds: 500),
                child: _statusAbsensi != null
                    ? Text(
                        _statusAbsensi!,
                        style: const TextStyle(
                            color: Colors.greenAccent, fontSize: 18),
                      )
                    : const SizedBox(),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
