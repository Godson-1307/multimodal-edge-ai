import React, { useState } from 'react';
import { 
  FileText, 
  ArrowLeft, 
  Download, 
  RotateCcw, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  Cpu, 
  Copy, 
  Check, 
  ExternalLink,
  Layers,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import { ExtractionResult, AppView } from '../types';

interface DocumentDetailsProps {
  document: ExtractionResult;
  onNavigate: (view: AppView) => void;
  onAnalyzeAgain: (doc: ExtractionResult) => void;
}

export const DocumentDetails: React.FC<DocumentDetailsProps> = ({
  document,
  onNavigate,
  onAnalyzeAgain,
}) => {
  const [copied, setCopied] = useState(false);
  const [zoom, setZoom] = useState(1);

  const jsonString = JSON.stringify(document.data, null, 2);

  const handleDownload = () => {
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${document.fileName.replace(/\.[^/.]+$/, '')}_details.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Top Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('history')}
            className="p-2 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-600 transition-colors"
            title="Back to History"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">
              Document Details
            </span>
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              {document.fileName}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={handleDownload}
            className="px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs border border-slate-200 shadow-xs flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download JSON</span>
          </button>

          <button
            onClick={() => onAnalyzeAgain(document)}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-sm flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Analyze Again</span>
          </button>
        </div>
      </div>

      {/* Metadata Bar (From prompt section 12: File, Uploaded, Status, Processing Time) */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
        <div>
          <span className="text-slate-400 block uppercase font-bold text-[10px]">File Name</span>
          <span className="font-bold text-slate-800 text-sm break-all">{document.fileName}</span>
          <span className="text-[11px] text-slate-400 font-mono block">{document.fileSize}</span>
        </div>

        <div>
          <span className="text-slate-400 block uppercase font-bold text-[10px]">Uploaded At</span>
          <span className="font-bold text-slate-800 text-sm">{document.uploadedAt}</span>
          <span className="text-[11px] text-slate-400 block">Local Edge Timestamp</span>
        </div>

        <div>
          <span className="text-slate-400 block uppercase font-bold text-[10px]">Status</span>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 mt-1 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-3 h-3" />
            {document.status}
          </span>
          <span className="text-[11px] text-slate-400 block mt-0.5">Quality: {document.qualityScore}%</span>
        </div>

        <div>
          <span className="text-slate-400 block uppercase font-bold text-[10px]">Processing Time</span>
          <span className="font-bold text-slate-900 text-sm font-mono">{document.processingTimeSec} sec</span>
          <span className="text-[11px] text-slate-400 block">{document.modelName.split(' ')[0]}</span>
        </div>
      </div>

      {/* Side by side layout for details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Original Document Image Section */}
        <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 shadow-sm p-4 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100 text-xs">
            <span className="font-bold text-slate-800 flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-blue-600" />
              Original Document
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setZoom(Math.max(0.8, zoom - 0.2))}
                className="p-1 hover:bg-slate-100 rounded text-slate-500"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setZoom(Math.min(1.6, zoom + 0.2))}
                className="p-1 hover:bg-slate-100 rounded text-slate-500"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="bg-slate-100 rounded-xl p-3 flex items-center justify-center overflow-auto max-h-[500px]">
            <img
              src={document.fileUrl}
              alt="Original Invoice"
              style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
              className="max-w-full h-auto rounded-lg shadow-sm border border-slate-200 transition-transform"
            />
          </div>
        </div>

        {/* Extracted Data Section */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-900">Extracted Data Summary</h3>
              <span className="text-[11px] font-mono text-emerald-600 font-bold bg-emerald-50 px-2 py-0.5 rounded">
                Validated
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Vendor Name:</span>
                <span className="font-bold text-slate-900">{document.data.vendor_name || 'N/A'}</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Invoice Number:</span>
                <span className="font-bold text-slate-900 font-mono">#{document.data.invoice_number || 'N/A'}</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Invoice Date:</span>
                <span className="font-semibold text-slate-800">{document.data.invoice_date || 'N/A'}</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">GSTIN:</span>
                <span className="font-bold text-blue-700 font-mono">{document.data.gstin || 'N/A'}</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">State / Code:</span>
                <span className="font-medium text-slate-800">
                  {document.data.state || 'N/A'} {document.data.state_code ? `(${document.data.state_code})` : ''}
                </span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Total Tax:</span>
                <span className="font-bold text-slate-800">
                  {document.data.total_tax !== null ? `${document.data.gstin?.startsWith('27') ? '₹ ' : '$ '}${document.data.total_tax.toLocaleString()}` : 'N/A'}
                </span>
              </div>

              <div className="flex justify-between py-2 bg-blue-50/50 px-3 rounded-xl border border-blue-100">
                <span className="font-bold text-blue-900">Total Amount:</span>
                <span className="font-extrabold text-blue-900 text-sm">
                  {document.data.total_amount !== null ? `${document.data.gstin?.startsWith('27') ? '₹ ' : '$ '}${document.data.total_amount.toLocaleString()}` : 'N/A'}
                </span>
              </div>
            </div>

            {/* Line items quick table */}
            {document.data.line_items.length > 0 && (
              <div className="pt-2">
                <span className="text-xs font-bold text-slate-800 block mb-2">
                  Line Items ({document.data.line_items.length})
                </span>
                <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                  {document.data.line_items.map((it, i) => (
                    <div key={i} className="flex justify-between text-xs p-2 bg-slate-50 rounded-lg">
                      <span className="font-medium text-slate-700">
                        {it.item} <span className="text-slate-400">×{it.quantity}</span>
                      </span>
                      <span className="font-bold text-slate-900">
                        {document.data.gstin?.startsWith('27') ? '₹ ' : '$ '}{it.price.toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Quick Raw JSON preview */}
          <div className="bg-slate-900 text-slate-200 rounded-2xl p-4 shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400 pb-2 border-b border-slate-800">
              <span className="font-mono">RAW JSON REPRESENTATION</span>
              <button
                onClick={handleCopy}
                className="text-cyan-400 hover:text-cyan-300 font-mono text-[11px] flex items-center gap-1"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'COPIED' : 'COPY'}</span>
              </button>
            </div>
            <pre className="text-emerald-400 font-mono text-[11px] max-h-40 overflow-y-auto leading-relaxed">
              {jsonString}
            </pre>
          </div>
        </div>

      </div>
    </div>
  );
};
