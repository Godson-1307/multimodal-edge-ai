import React from 'react';
import { 
  FileText, 
  Cpu, 
  ShieldCheck, 
  Zap, 
  Braces, 
  ArrowRight, 
  CheckCircle, 
  Layers, 
  Sparkles
} from 'lucide-react';
import { AppView } from '../types';

interface LandingPageProps {
  onNavigate: (view: AppView) => void;
  onLoginClick: () => void;
  onTrySample: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onNavigate,
  onLoginClick,
  onTrySample,
}) => {
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-20 lg:pt-20 lg:pb-28 border-b border-slate-200/80 bg-gradient-to-b from-white via-slate-50/50 to-slate-100/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            
            {/* Left Copy */}
            <div className="lg:col-span-7 space-y-6 text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-200/80 text-blue-700 text-xs font-semibold">
                <span className="flex h-2 w-2 rounded-full bg-blue-600 animate-ping"></span>
                <span>Powered by Qwen2-VL &amp; QLoRA Fine-Tuning</span>
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.12]">
                AI-Powered Document Understanding{' '}
                <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-600 bg-clip-text text-transparent">
                  at the Edge.
                </span>
              </h1>

              <p className="text-lg sm:text-xl text-slate-600 max-w-2xl leading-relaxed">
                Extract key-value fields, line items, and structured JSON from invoices and receipts in seconds. 
                Runs privately on local edge hardware with high-precision Vision-Language intelligence.
              </p>

              {/* CTAs */}
              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  id="hero-get-started-btn"
                  onClick={() => onNavigate('upload')}
                  className="px-6 py-3.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-base shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/30 transition-all flex items-center gap-2"
                >
                  <span>Upload Document</span>
                  <ArrowRight className="w-4 h-4" />
                </button>

                <button
                  id="hero-try-sample-btn"
                  onClick={onTrySample}
                  className="px-6 py-3.5 rounded-xl bg-white hover:bg-slate-100 text-slate-800 font-semibold text-base border border-slate-300 shadow-sm transition-all flex items-center gap-2"
                >
                  <Sparkles className="w-4 h-4 text-blue-600" />
                  <span>Try Sample Invoice</span>
                </button>

                <button
                  id="hero-login-btn"
                  onClick={onLoginClick}
                  className="px-4 py-3.5 text-sm font-medium text-slate-600 hover:text-slate-900"
                >
                  Existing User? Login
                </button>
              </div>

              {/* Edge AI Highlights pill */}
              <div className="pt-4 flex flex-wrap items-center gap-6 text-xs text-slate-600 font-medium">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                  <span>CORD Dataset Benchmark</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                  <span>100% Local Inference Support</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                  <span>Sub-2.5s Latency</span>
                </div>
              </div>
            </div>

            {/* Right Mock Visual (Side-by-side Edge Extraction Preview) */}
            <div className="lg:col-span-5">
              <div className="relative rounded-2xl bg-white p-4 shadow-xl shadow-slate-200/70 border border-slate-200">
                {/* Visual Header */}
                <div className="flex items-center justify-between pb-3 border-b border-slate-100 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-rose-400"></span>
                    <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                    <span className="w-3 h-3 rounded-full bg-emerald-400"></span>
                    <span className="font-semibold text-slate-700 ml-1">Live Extraction Stream</span>
                  </div>
                  <span className="font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded text-[11px] font-bold">
                    99.4% CONFIDENCE
                  </span>
                </div>

                {/* Simulated Document + Extraction preview */}
                <div className="py-4 space-y-3">
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/80">
                    <div className="flex items-start justify-between text-xs text-slate-500 mb-2">
                      <span className="font-mono">invoice_topstar_725.jpg</span>
                      <span className="text-blue-600 font-semibold">Qwen2-VL-2B</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-white p-2 rounded border border-slate-100">
                        <span className="text-[10px] text-slate-400 uppercase font-bold block">Vendor</span>
                        <span className="font-bold text-slate-800">TOP STAR NAMKEEN</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-slate-100">
                        <span className="text-[10px] text-slate-400 uppercase font-bold block">Invoice #</span>
                        <span className="font-bold text-slate-800">#725</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-slate-100">
                        <span className="text-[10px] text-slate-400 uppercase font-bold block">GSTIN</span>
                        <span className="font-mono font-medium text-slate-800 text-[11px]">27AGFPN1247B1ZS</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-slate-100">
                        <span className="text-[10px] text-slate-400 uppercase font-bold block">Total Amount</span>
                        <span className="font-bold text-blue-700">₹ 15,91,600</span>
                      </div>
                    </div>
                  </div>

                  {/* Line items snippet */}
                  <div className="bg-slate-900 text-slate-200 p-3 rounded-xl font-mono text-[11px] space-y-1 overflow-hidden">
                    <div className="flex justify-between text-slate-400 text-[10px] border-b border-slate-800 pb-1">
                      <span>STRUCTURED JSON OUTPUT</span>
                      <span>STATUS: 200 OK</span>
                    </div>
                    <pre className="text-emerald-400 pt-1 leading-tight">
{`{
  "vendor_name": "TOP STAR NAMKEEN",
  "invoice_number": "725",
  "total_amount": 1591600,
  "line_items_count": 3
}`}
                    </pre>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
                  <span className="flex items-center gap-1">
                    <Cpu className="w-3.5 h-3.5 text-blue-600" /> Edge GPU inference: 2.14s
                  </span>
                  <button 
                    onClick={() => onNavigate('upload')}
                    className="text-blue-600 hover:text-blue-700 font-semibold"
                  >
                    View Full Schema →
                  </button>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Features Section (Direct translation from user prompt) */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-xs font-bold uppercase tracking-widest text-blue-600 mb-2">Core Capabilities</h2>
            <p className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
              Engineered for Real-World Document Intelligence
            </p>
            <p className="text-base text-slate-600 mt-3">
              Combining state-of-the-art vision encoders with parameter-efficient fine-tuning for lightweight edge deployment.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            
            {/* 1. Document OCR */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center mb-5">
                <FileText className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                📄 Document OCR
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Extract high-fidelity information from invoices, dining receipts, freight slips, and bills with zero optical noise.
              </p>
            </div>

            {/* 2. AI Understanding */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center mb-5">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                🤖 AI Understanding
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Vision-Language Model (Qwen2-VL) grasps spatial layout, multi-column tables, headers, and semantic relationships without brittle regex templates.
              </p>
            </div>

            {/* 3. Privacy */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center mb-5">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                🔒 Privacy First
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Documents can be processed entirely locally on the edge. Sensitive financial figures and GST numbers never leave your private perimeter.
              </p>
            </div>

            {/* 4. Fast Processing */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center mb-5">
                <Zap className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                ⚡ Fast Processing
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                GPU-accelerated 4-bit quantized inference delivers end-to-end token generation in under 2.5 seconds on embedded chips and consumer cards.
              </p>
            </div>

            {/* 5. Structured Output */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-cyan-100 text-cyan-700 flex items-center justify-center mb-5">
                <Braces className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                📊 Structured Output
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Get cleanly normalized JSON data with validated schemas, complete line-item breakdown, and field confidence scoring ready for ERP export.
              </p>
            </div>

            {/* 6. Fine-Tuned QLoRA */}
            <div className="p-7 rounded-2xl bg-slate-50/70 border border-slate-200/80 hover:border-blue-300 hover:shadow-md transition-all">
              <div className="w-12 h-12 rounded-xl bg-rose-100 text-rose-700 flex items-center justify-center mb-5">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                🧠 Fine-Tuned QLoRA
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Trained specifically on CORD receipts, tax invoices, and varied real-world receipt typography for superior extraction accuracy.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* Call To Action Banner */}
      <section className="py-16 bg-slate-100/60 border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-950 rounded-2xl p-8 sm:p-12 text-white flex flex-col md:flex-row items-center justify-between gap-6 shadow-xl">
            <div className="space-y-2 text-center md:text-left">
              <h3 className="text-2xl sm:text-3xl font-bold tracking-tight">Ready to test Edge Document OCR?</h3>
              <p className="text-slate-300 text-sm max-w-xl">
                Upload your invoice or test immediately with pre-loaded samples like TOP STAR NAMKEEN or CORD benchmarks.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                id="cta-analyze-now-btn"
                onClick={() => onNavigate('upload')}
                className="px-6 py-3.5 rounded-xl bg-blue-500 hover:bg-blue-600 text-white font-semibold text-sm shadow-md transition-all flex items-center gap-2"
              >
                <span>Analyze Document Now</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2 font-medium text-slate-700">
            <div className="w-5 h-5 rounded bg-blue-600 text-white flex items-center justify-center font-bold text-[10px]">
              E
            </div>
            <span>Multi-Modal Edge AI — Document OCR &amp; Invoice Extraction</span>
          </div>
          <p>Qwen2-VL + QLoRA Model Architecture • College Demonstration &amp; Viva Prototype</p>
        </div>
      </footer>
    </div>
  );
};
