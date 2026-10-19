import React from 'react';
import { 
  CheckCircle2, 
  Loader2, 
  Clock, 
  Cpu, 
  FileText, 
  Sparkles, 
  Layers, 
  ShieldCheck 
} from 'lucide-react';
import { PIPELINE_STEPS } from '../mockData';

interface ProcessingViewProps {
  currentStageIndex: number;
  fileName: string;
}

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  currentStageIndex,
  fileName,
}) => {
  return (
    <div className="max-w-2xl mx-auto px-4 py-16">
      <div className="bg-white rounded-3xl border border-slate-200 shadow-xl shadow-slate-200/50 p-8 sm:p-12 text-center space-y-8">
        
        {/* Animated Scanner Ring */}
        <div className="relative w-24 h-24 mx-auto flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-4 border-blue-100 animate-ping opacity-60"></div>
          <div className="absolute inset-0 rounded-full border-4 border-t-blue-600 border-r-indigo-600 border-b-cyan-500 border-l-transparent animate-spin"></div>
          <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-700 text-white flex items-center justify-center shadow-lg shadow-blue-500/30">
            <Cpu className="w-8 h-8 animate-pulse" />
          </div>
        </div>

        {/* Heading */}
        <div className="space-y-2">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Analyzing your document...
          </h2>
          <p className="text-sm text-slate-500 flex items-center justify-center gap-2">
            <FileText className="w-4 h-4 text-slate-400" />
            <span className="font-semibold text-slate-700 font-mono">{fileName}</span>
          </p>
        </div>

        {/* Step-by-step pipeline status checklist (Matching prompt requirements) */}
        <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200/80 text-left space-y-4">
          <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider pb-2 border-b border-slate-200">
            <span>Inference Pipeline Stage</span>
            <span>Status</span>
          </div>

          <div className="space-y-3.5">
            {PIPELINE_STEPS.map((step, idx) => {
              const isCompleted = idx < currentStageIndex;
              const isActive = idx === currentStageIndex;
              const isPending = idx > currentStageIndex;

              return (
                <div
                  key={step.id}
                  className={`flex items-center justify-between p-3 rounded-xl transition-all ${
                    isActive
                      ? 'bg-blue-50/90 border border-blue-200 text-blue-950 font-semibold shadow-xs'
                      : isCompleted
                      ? 'text-slate-800'
                      : 'text-slate-400 opacity-60'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="shrink-0">
                      {isCompleted && (
                        <div className="w-6 h-6 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
                          <CheckCircle2 className="w-4 h-4" />
                        </div>
                      )}
                      {isActive && (
                        <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center">
                          <Loader2 className="w-4 h-4 animate-spin" />
                        </div>
                      )}
                      {isPending && (
                        <div className="w-6 h-6 rounded-full bg-slate-200 text-slate-400 flex items-center justify-center text-xs font-bold">
                          {idx + 1}
                        </div>
                      )}
                    </div>

                    <div>
                      <div className="text-xs sm:text-sm font-bold flex items-center gap-2">
                        {step.name}
                        {isActive && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-600 text-white font-normal uppercase tracking-wider animate-pulse">
                            Processing
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-500 hidden sm:block">
                        {step.description}
                      </div>
                    </div>
                  </div>

                  <div className="text-xs font-mono shrink-0 pl-2">
                    {isCompleted && <span className="text-emerald-600 font-bold">✓ Done</span>}
                    {isActive && <span className="text-blue-600 font-bold">🔄 Active</span>}
                    {isPending && <span className="text-slate-400">Waiting...</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Warning / Edge telemetry note */}
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200 text-xs font-medium">
            <Clock className="w-3.5 h-3.5 text-amber-600 animate-spin" />
            <span>Processing on edge device... Please don't close this page.</span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-500 pt-2 border-t border-slate-100 font-mono">
            <div>
              <span className="text-slate-400 block text-[10px]">BACKBONE</span>
              <span className="font-semibold text-slate-700">Qwen2-VL-2B</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px]">QUANTIZATION</span>
              <span className="font-semibold text-slate-700">INT4 AWQ</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px]">DEVICE</span>
              <span className="font-semibold text-slate-700">Edge GPU Node</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
