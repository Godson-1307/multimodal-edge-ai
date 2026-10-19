import React, { useState } from 'react';
import { 
  FileText, 
  UploadCloud, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  TrendingUp, 
  Sparkles, 
  ArrowUpRight, 
  Search, 
  Filter,
  MoreVertical,
  Cpu,
  Eye,
  Layers
} from 'lucide-react';
import { ExtractionResult, AppView, UserProfile } from '../types';

interface DashboardProps {
  user: UserProfile | null;
  history: ExtractionResult[];
  onNavigate: (view: AppView) => void;
  onSelectDocument: (doc: ExtractionResult) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  user,
  history,
  onNavigate,
  onSelectDocument,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'All' | 'Completed' | 'Processing' | 'Failed'>('All');

  // Compute live statistics
  const totalProcessed = history.length;
  const successful = history.filter((d) => d.status === 'Completed').length;
  const failed = history.filter((d) => d.status === 'Failed').length;
  const processing = history.filter((d) => d.status === 'Processing').length;

  // Average AI Quality score
  const avgQuality = Math.round(
    history.reduce((acc, curr) => acc + (curr.qualityScore || 0), 0) / (history.length || 1)
  );

  // Filtered documents
  const filteredDocs = history.filter((doc) => {
    const matchesSearch =
      doc.fileName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (doc.data.vendor_name && doc.data.vendor_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (doc.data.invoice_number && doc.data.invoice_number.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesStatus = statusFilter === 'All' ? true : doc.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const userName = user?.name ? user.name.split(' ')[0] : 'Sarah';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Welcome Banner & Top Quick Action (Inspired by uploaded screenshot & prompt) */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200/90 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Welcome back, {userName}! 👋
            </h1>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Summary for edge document OCR, invoice parsing, and Qwen2-VL extraction pipeline.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            id="dashboard-upload-btn"
            onClick={() => onNavigate('upload')}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm shadow-sm shadow-blue-500/20 hover:shadow-md transition-all flex items-center gap-2"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload Document</span>
          </button>

          <button
            id="dashboard-automate-btn"
            onClick={() => onNavigate('upload')}
            className="px-4 py-2.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 text-sm font-semibold transition-all flex items-center gap-1.5"
          >
            <Sparkles className="w-4 h-4 text-cyan-600" />
            <span>Automate with AI</span>
          </button>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        
        {/* Total Documents Processed */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>Documents Processed</span>
            <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
              <FileText className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900">{totalProcessed}</span>
            <span className="text-xs font-semibold text-emerald-600 flex items-center">
              +15% this week
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Total OCR jobs executed</p>
        </div>

        {/* Successful Extractions */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>Successful</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <CheckCircle2 className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-emerald-700">{successful}</span>
            <span className="text-xs font-medium text-slate-500">
              ({Math.round((successful / (totalProcessed || 1)) * 100)}% pass rate)
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Validated schema outputs</p>
        </div>

        {/* Failed Extractions */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>Failed / Low Res</span>
            <div className="w-8 h-8 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
              <AlertCircle className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-rose-600">{failed}</span>
            <span className="text-xs font-medium text-slate-400">
              {failed > 0 ? 'Requires clearer scan' : 'Clean stream'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Blurred or cropped input</p>
        </div>

        {/* AI Quality & Insights */}
        <div className="bg-gradient-to-br from-slate-900 to-indigo-950 text-white rounded-2xl p-5 shadow-sm border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
            <span className="flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              AI Extraction Score
            </span>
            <span className="px-2 py-0.5 rounded-full bg-cyan-400/20 text-cyan-300 font-mono text-[10px] font-bold">
              Qwen2-VL
            </span>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white">{avgQuality}%</span>
            <span className="text-xs font-medium text-cyan-400">Fine-Tuned QLoRA</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Average confidence across GSTIN &amp; line items</p>
        </div>

      </div>

      {/* AI Invoice Insights Banner (From user's screenshot layout) */}
      <div className="bg-blue-50/60 border border-blue-200/80 rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="w-9 h-9 rounded-xl bg-blue-600 text-white flex items-center justify-center shrink-0">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-900">AI Edge Intelligence Tips</h4>
            <p className="text-xs text-slate-600 mt-0.5">
              Invoice <span className="font-semibold text-slate-900">#INV-0254</span> (Vertex Solutions) &amp; <span className="font-semibold text-slate-900">TOP STAR NAMKEEN #725</span> both scored 99%+ extraction confidence. QLoRA adapters resolved multi-line items accurately.
            </p>
          </div>
        </div>
        <button
          onClick={() => onNavigate('how-it-works')}
          className="text-xs font-bold text-blue-700 hover:text-blue-900 whitespace-nowrap flex items-center gap-1"
        >
          <span>View Model Metrics</span>
          <ArrowUpRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Recent Documents Table Section */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {/* Table Header Controls */}
        <div className="p-5 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-slate-900">Recent Documents</h3>
            <p className="text-xs text-slate-500">Invoices and receipts analyzed through the edge pipeline</p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            {/* Search input */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search invoices..."
                className="pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 w-48 sm:w-56"
              />
            </div>

            {/* Filter Pills */}
            <div className="flex items-center bg-slate-100 p-1 rounded-lg text-xs font-medium text-slate-600">
              {(['All', 'Completed', 'Failed'] as const).map((filter) => (
                <button
                  key={filter}
                  onClick={() => setStatusFilter(filter)}
                  className={`px-3 py-1 rounded-md transition-all ${
                    statusFilter === filter
                      ? 'bg-white text-slate-900 shadow-sm font-semibold'
                      : 'hover:text-slate-900'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50/80 border-b border-slate-200/80 text-slate-500 font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4 w-12 text-center">#</th>
                <th className="py-3 px-4">Document File</th>
                <th className="py-3 px-4">Vendor / Client</th>
                <th className="py-3 px-4">Upload Date</th>
                <th className="py-3 px-4 text-right">Amount</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4 text-center">AI Score</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredDocs.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-400">
                    No documents found matching your filter.
                  </td>
                </tr>
              ) : (
                filteredDocs.map((doc, idx) => {
                  const isCompleted = doc.status === 'Completed';
                  const isFailed = doc.status === 'Failed';
                  const isProcessing = doc.status === 'Processing';

                  return (
                    <tr
                      key={doc.id}
                      className="hover:bg-slate-50/80 transition-colors group cursor-pointer"
                      onClick={() => onSelectDocument(doc)}
                    >
                      <td className="py-3.5 px-4 text-center text-slate-400 font-mono">
                        {idx + 1}
                      </td>

                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-2.5">
                          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center shrink-0 border border-blue-100">
                            <FileText className="w-4 h-4" />
                          </div>
                          <div>
                            <span className="font-semibold text-slate-900 block group-hover:text-blue-600 transition-colors">
                              {doc.fileName}
                            </span>
                            <span className="text-[11px] text-slate-400 font-mono">
                              {doc.fileSize} • {doc.processingTimeSec}s
                            </span>
                          </div>
                        </div>
                      </td>

                      <td className="py-3.5 px-4 font-medium text-slate-800">
                        {doc.data.vendor_name || <span className="text-slate-400 italic">Not detected</span>}
                        {doc.data.invoice_number && (
                          <span className="block text-[11px] text-slate-400">#{doc.data.invoice_number}</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4 text-slate-500">
                        {doc.uploadedAt}
                      </td>

                      <td className="py-3.5 px-4 text-right font-bold text-slate-900">
                        {doc.data.total_amount !== null ? (
                          <span>
                            {doc.data.gstin?.startsWith('27') ? '₹ ' : '$ '}
                            {doc.data.total_amount.toLocaleString()}
                          </span>
                        ) : (
                          <span className="text-slate-400 font-normal">-</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4 text-center">
                        <span
                          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-semibold ${
                            isCompleted
                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                              : isFailed
                              ? 'bg-rose-50 text-rose-700 border border-rose-200'
                              : 'bg-amber-50 text-amber-700 border border-amber-200 animate-pulse'
                          }`}
                        >
                          {isCompleted && <CheckCircle2 className="w-3 h-3" />}
                          {isFailed && <AlertCircle className="w-3 h-3" />}
                          {isProcessing && <Clock className="w-3 h-3" />}
                          {doc.status}
                        </span>
                      </td>

                      <td className="py-3.5 px-4 text-center">
                        {doc.qualityScore > 0 ? (
                          <span className="inline-block px-2 py-0.5 rounded-md bg-cyan-50 text-cyan-800 font-bold border border-cyan-200/60 font-mono text-[11px]">
                            {doc.qualityScore}%
                          </span>
                        ) : (
                          <span className="text-slate-400">-</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectDocument(doc);
                          }}
                          className="px-3 py-1 bg-slate-100 hover:bg-blue-600 hover:text-white rounded-lg text-slate-700 font-medium text-xs transition-colors inline-flex items-center gap-1"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Table Footer */}
        <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-slate-50/50">
          <span>Showing {filteredDocs.length} of {history.length} documents</span>
          <button
            onClick={() => onNavigate('history')}
            className="text-blue-600 hover:text-blue-800 font-semibold"
          >
            View Full History →
          </button>
        </div>
      </div>
    </div>
  );
};
