import React, { useState } from 'react';
import { 
  FileText, 
  Search, 
  Filter, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  Eye, 
  Download, 
  ArrowLeft,
  UploadCloud,
  Cpu,
  Trash2
} from 'lucide-react';
import { ExtractionResult, AppView } from '../types';

interface DocumentHistoryProps {
  history: ExtractionResult[];
  onSelectDocument: (doc: ExtractionResult) => void;
  onNavigate: (view: AppView) => void;
  onClearHistory?: () => void;
}

export const DocumentHistory: React.FC<DocumentHistoryProps> = ({
  history,
  onSelectDocument,
  onNavigate,
  onClearHistory,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'All' | 'Completed' | 'Processing' | 'Failed'>('All');

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.fileName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (item.data.vendor_name && item.data.vendor_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (item.data.invoice_number && item.data.invoice_number.toLowerCase().includes(searchTerm.toLowerCase()));

    const matchesStatus = statusFilter === 'All' ? true : item.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onNavigate('dashboard')}
              className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-100 text-slate-500"
              title="Back"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Document History
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Archived documents processed through edge OCR inference pipeline
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('upload')}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-sm flex items-center gap-1.5"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload New Document</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Filter Pills (Prompt specified: [ All ], [ Successful ], [ Failed ], [ Processing ]) */}
        <div className="flex items-center bg-slate-100 p-1 rounded-xl text-xs font-semibold text-slate-600 overflow-x-auto">
          {(['All', 'Completed', 'Processing', 'Failed'] as const).map((filter) => {
            const label = filter === 'Completed' ? 'Successful' : filter;
            const count = filter === 'All' ? history.length : history.filter((h) => h.status === filter).length;

            return (
              <button
                key={filter}
                onClick={() => setStatusFilter(filter)}
                className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 whitespace-nowrap ${
                  statusFilter === filter
                    ? 'bg-white text-slate-900 shadow-sm font-bold'
                    : 'hover:text-slate-900'
                }`}
              >
                <span>{label}</span>
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-200 text-slate-700">
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by filename, vendor, #..."
            className="pl-9 pr-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 w-full sm:w-64 text-slate-900"
          />
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-3.5 px-4 w-12 text-center">#</th>
                <th className="py-3.5 px-4">Document</th>
                <th className="py-3.5 px-4">Vendor Name</th>
                <th className="py-3.5 px-4">Date</th>
                <th className="py-3.5 px-4 text-center">Latency</th>
                <th className="py-3.5 px-4 text-center">Status</th>
                <th className="py-3.5 px-4 text-center">AI Quality</th>
                <th className="py-3.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-16 text-center text-slate-400">
                    <FileText className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                    <p className="font-semibold text-slate-600">No documents found</p>
                    <p className="text-xs text-slate-400">Try adjusting your filters or upload a new invoice</p>
                  </td>
                </tr>
              ) : (
                filteredHistory.map((item, idx) => (
                  <tr
                    key={item.id}
                    onClick={() => onSelectDocument(item)}
                    className="hover:bg-slate-50/80 transition-colors cursor-pointer group"
                  >
                    <td className="py-4 px-4 text-center text-slate-400 font-mono">
                      {idx + 1}
                    </td>

                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0 border border-blue-100">
                          <FileText className="w-4 h-4" />
                        </div>
                        <div>
                          <span className="font-bold text-slate-900 block group-hover:text-blue-600 transition-colors">
                            {item.fileName}
                          </span>
                          <span className="text-[11px] text-slate-400 font-mono">
                            {item.fileSize} • {item.fileType}
                          </span>
                        </div>
                      </div>
                    </td>

                    <td className="py-4 px-4 font-semibold text-slate-800">
                      {item.data.vendor_name || <span className="text-slate-400 italic">Not detected</span>}
                      {item.data.invoice_number && (
                        <span className="block text-[11px] text-slate-400 font-mono">
                          #{item.data.invoice_number}
                        </span>
                      )}
                    </td>

                    <td className="py-4 px-4 text-slate-500 font-medium">
                      {item.uploadedAt}
                    </td>

                    <td className="py-4 px-4 text-center font-mono text-slate-600">
                      {item.processingTimeSec}s
                    </td>

                    <td className="py-4 px-4 text-center">
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold ${
                          item.status === 'Completed'
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : item.status === 'Failed'
                            ? 'bg-rose-50 text-rose-700 border border-rose-200'
                            : 'bg-amber-50 text-amber-700 border border-amber-200 animate-pulse'
                        }`}
                      >
                        {item.status === 'Completed' && <CheckCircle2 className="w-3 h-3" />}
                        {item.status === 'Failed' && <AlertCircle className="w-3 h-3" />}
                        {item.status === 'Processing' && <Clock className="w-3 h-3" />}
                        <span>{item.status === 'Completed' ? 'Completed' : item.status}</span>
                      </span>
                    </td>

                    <td className="py-4 px-4 text-center">
                      {item.qualityScore > 0 ? (
                        <span className="inline-block px-2 py-0.5 rounded-md bg-cyan-50 text-cyan-800 font-bold border border-cyan-200 font-mono text-[11px]">
                          {item.qualityScore}%
                        </span>
                      ) : (
                        <span className="text-slate-400 font-mono">-</span>
                      )}
                    </td>

                    <td className="py-4 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectDocument(item);
                        }}
                        className="px-3.5 py-1.5 bg-slate-100 hover:bg-blue-600 hover:text-white rounded-lg text-slate-700 font-semibold text-xs transition-colors inline-flex items-center gap-1"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>View</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Footer info */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
          <span>Total records: {history.length}</span>
          <span className="font-mono">Engine: Qwen2-VL-2B (Edge INT4)</span>
        </div>
      </div>
    </div>
  );
};
