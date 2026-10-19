import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  X, 
  Sparkles, 
  ArrowRight, 
  Cpu, 
  CheckCircle2, 
  AlertCircle,
  FileSpreadsheet,
  Image as ImageIcon,
  Zap
} from 'lucide-react';
import { MOCK_INVOICE_TOPSTAR, MOCK_INVOICE_CORD, MOCK_INVOICE_VERTEX } from '../mockData';
import { EdgeSettings } from '../types';

interface UploadDocumentProps {
  onStartAnalysis: (file: File | { name: string; size: number; type: string; dataUrl?: string }) => void;
  settings: EdgeSettings;
}

export const UploadDocument: React.FC<UploadDocumentProps> = ({ onStartAnalysis, settings }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<{
    file?: File;
    name: string;
    size: string;
    type: string;
    previewUrl?: string;
  } | null>(null);

  const [selectedModel, setSelectedModel] = useState<'qwen2-vl-2b' | 'qwen2-vl-7b'>('qwen2-vl-2b');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const processFile = (file: File) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|pdf)$/i)) {
      alert('Supported files: PDF, JPG, JPEG, PNG');
      return;
    }

    const sizeStr =
      file.size > 1024 * 1024
        ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
        : `${Math.round(file.size / 1024)} KB`;

    let previewUrl = '';
    if (file.type.startsWith('image/')) {
      previewUrl = URL.createObjectURL(file);
    }

    setSelectedFile({
      file,
      name: file.name,
      size: sizeStr,
      type: file.type,
      previewUrl,
    });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleSelectSample = (sample: typeof MOCK_INVOICE_TOPSTAR) => {
    setSelectedFile({
      name: sample.fileName,
      size: sample.fileSize,
      type: sample.fileType,
      previewUrl: sample.fileUrl,
    });
  };

  const handleTriggerAnalysis = () => {
    if (!selectedFile) return;
    if (selectedFile.file) {
      onStartAnalysis(selectedFile.file);
    } else {
      // Sample mock file
      onStartAnalysis({
        name: selectedFile.name,
        size: 2400000,
        type: selectedFile.type,
        dataUrl: selectedFile.previewUrl,
      });
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      {/* Page Heading */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Upload Document</h1>
        <p className="text-slate-600 text-sm max-w-xl mx-auto">
          Upload an invoice or receipt image/PDF for edge OCR extraction powered by Qwen2-VL &amp; QLoRA.
        </p>
      </div>

      {/* Main Upload Box */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
        
        {!selectedFile ? (
          /* Drag and Drop Zone */
          <div
            id="upload-dropzone"
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 sm:p-14 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-4 ${
              dragActive
                ? 'border-blue-500 bg-blue-50/50 scale-[0.99]'
                : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50/70 bg-slate-50/30'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,image/jpeg,image/png,application/pdf"
              className="hidden"
              onChange={handleFileChange}
            />

            <div className="w-16 h-16 rounded-2xl bg-blue-100/70 text-blue-600 flex items-center justify-center shadow-sm">
              <FileText className="w-8 h-8" />
            </div>

            <div className="space-y-1">
              <p className="text-base font-bold text-slate-800">
                Drag &amp; Drop Document
              </p>
              <p className="text-xs text-slate-500">
                Drop your file directly here, or
              </p>
            </div>

            <button
              id="choose-file-btn"
              type="button"
              className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-sm transition-all flex items-center gap-1.5"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Choose File</span>
            </button>

            <div className="pt-2 text-[11px] text-slate-400 font-medium">
              Supported files: <span className="font-semibold text-slate-600">PDF, JPG, JPEG, PNG</span> (Max 25MB)
            </div>
          </div>
        ) : (
          /* Selected File Card */
          <div className="space-y-6">
            <div className="p-4 sm:p-5 rounded-xl bg-slate-50 border border-slate-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                {selectedFile.previewUrl ? (
                  <img
                    src={selectedFile.previewUrl}
                    alt="Preview"
                    className="w-16 h-20 object-cover rounded-lg border border-slate-200 shadow-sm bg-white"
                  />
                ) : (
                  <div className="w-16 h-20 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center border border-blue-200">
                    <FileText className="w-8 h-8" />
                  </div>
                )}
                <div>
                  <span className="text-[11px] font-bold text-blue-600 uppercase tracking-wider block">
                    Selected File Ready for Inference
                  </span>
                  <h3 className="text-base font-bold text-slate-900 break-all">{selectedFile.name}</h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {selectedFile.size} • {selectedFile.type || 'Document'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2 self-end sm:self-center">
                <button
                  id="remove-file-btn"
                  onClick={() => setSelectedFile(null)}
                  className="px-3.5 py-2 rounded-lg text-xs font-semibold text-slate-600 hover:text-rose-600 hover:bg-rose-50 border border-slate-200 transition-colors flex items-center gap-1"
                >
                  <X className="w-3.5 h-3.5" />
                  <span>Remove</span>
                </button>
              </div>
            </div>

            {/* Model & Edge Target Options */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3.5 rounded-xl bg-white border border-slate-200">
                <label className="font-bold text-slate-800 block mb-1.5 flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-blue-600" />
                  Model Configuration
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as any)}
                  className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs font-medium text-slate-800"
                >
                  <option value="qwen2-vl-2b">Qwen2-VL-2B (QLoRA 4-bit Edge Optimized)</option>
                  <option value="qwen2-vl-7b">Qwen2-VL-7B (High-Precision Full Attention)</option>
                </select>
                <p className="text-[11px] text-slate-400 mt-1">Recommended for sub-2.5s edge inference</p>
              </div>

              <div className="p-3.5 rounded-xl bg-white border border-slate-200">
                <label className="font-bold text-slate-800 block mb-1.5 flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-emerald-600" />
                  Extraction Engine
                </label>
                <div className="p-2 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between">
                  <span className="font-medium text-slate-700">
                    Vision-Language Model (Ready)
                  </span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Automatic key-value and tabular recognition</p>
              </div>
            </div>

            {/* Analyze CTA */}
            <button
              id="analyze-document-btn"
              onClick={handleTriggerAnalysis}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold text-sm shadow-md shadow-blue-500/25 transition-all flex items-center justify-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              <span>Analyze Document</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}

      </div>

      {/* One-Click Benchmark Sample Invoices (Great for viva testing!) */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-blue-600" />
              One-Click Sample Documents
            </h3>
            <p className="text-xs text-slate-500">Try with real test invoices from the user prompt &amp; CORD benchmark dataset</p>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-mono">
            3 Ready Presets
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          
          {/* Sample 1: TOP STAR NAMKEEN */}
          <button
            id="sample-topstar-btn"
            onClick={() => handleSelectSample(MOCK_INVOICE_TOPSTAR)}
            className="p-3 text-left rounded-xl border border-slate-200 hover:border-blue-400 hover:bg-blue-50/40 transition-all group"
          >
            <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
              <span className="font-mono">GST INVOICE</span>
              <span className="text-blue-600 font-bold">₹ 15.9L</span>
            </div>
            <h4 className="font-bold text-slate-900 text-xs group-hover:text-blue-600 transition-colors">
              TOP STAR NAMKEEN
            </h4>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Invoice #725 • 3 Line items • GSTIN
            </p>
          </button>

          {/* Sample 2: CORD Bali Restaurant */}
          <button
            id="sample-cord-btn"
            onClick={() => handleSelectSample(MOCK_INVOICE_CORD)}
            className="p-3 text-left rounded-xl border border-slate-200 hover:border-blue-400 hover:bg-blue-50/40 transition-all group"
          >
            <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
              <span className="font-mono">CORD RECEIPT</span>
              <span className="text-blue-600 font-bold">IDR 429k</span>
            </div>
            <h4 className="font-bold text-slate-900 text-xs group-hover:text-blue-600 transition-colors">
              Bebek Bengil Resto &amp; Cafe
            </h4>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Nasi Campur Bali • CORD target
            </p>
          </button>

          {/* Sample 3: Vertex Solutions */}
          <button
            id="sample-vertex-btn"
            onClick={() => handleSelectSample(MOCK_INVOICE_VERTEX)}
            className="p-3 text-left rounded-xl border border-slate-200 hover:border-blue-400 hover:bg-blue-50/40 transition-all group"
          >
            <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
              <span className="font-mono">TECH INVOICE</span>
              <span className="text-blue-600 font-bold">$ 12,500</span>
            </div>
            <h4 className="font-bold text-slate-900 text-xs group-hover:text-blue-600 transition-colors">
              Vertex Solutions Inc.
            </h4>
            <p className="text-[11px] text-slate-500 mt-0.5">
              INV-0254 • Edge AI Server Nodes
            </p>
          </button>

        </div>
      </div>
    </div>
  );
};
