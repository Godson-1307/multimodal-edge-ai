import React, { useState } from 'react';

import {
  FileText,
  Copy,
  Check,
  Download,
  ArrowLeft,
  ZoomIn,
  ZoomOut,
  CheckCircle2,
  AlertTriangle,
  Table,
  Code2,
  Eye,
  EyeOff,
  RotateCcw,
  Clock,
} from 'lucide-react';

import { ExtractionResult, AppView } from '../types';

interface ExtractionResultsProps {
  result: ExtractionResult;
  onNavigate: (view: AppView) => void;
  onAnalyzeAnother: () => void;
}

export const ExtractionResults: React.FC<ExtractionResultsProps> = ({
  result,
  onNavigate,
  onAnalyzeAnother,
}) => {
  const [activeTab, setActiveTab] = useState<'fields' | 'line_items' | 'json'>(
    'fields'
  );

  const [copiedJson, setCopiedJson] = useState(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeHoverField, setActiveHoverField] = useState<string | null>(null);

  const { data, confidences } = result;

  /*
   * Count fields that actually contain extracted values.
   * This is a detection count, NOT an accuracy/confidence score.
   */
  const fieldValues = [
    data.vendor_name,
    data.invoice_number,
    data.invoice_date,
    data.gstin,
    data.state,
    data.state_code,
    data.total_tax,
    data.total_amount,
  ];

  const detectedFieldsCount = fieldValues.filter(
    (value) => value !== null && value !== undefined && value !== ''
  ).length;

  /*
   * JSON output
   */
  const formattedJsonString = JSON.stringify(
    {
      vendor_name: data.vendor_name,
      invoice_number: data.invoice_number,
      invoice_date: data.invoice_date,
      gstin: data.gstin,
      state: data.state,
      state_code: data.state_code,
      total_tax: data.total_tax,
      total_amount: data.total_amount,
      line_items: data.line_items,
      model_metadata: {
        model: result.modelName,
        processing_time_sec: result.processingTimeSec,
      },
    },
    null,
    2
  );

  const handleCopyJson = () => {
    navigator.clipboard.writeText(formattedJsonString);

    setCopiedJson(true);

    setTimeout(() => {
      setCopiedJson(false);
    }, 2000);
  };

  const handleDownloadJson = () => {
    const blob = new Blob([formattedJsonString], {
      type: 'application/json',
    });

    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');

    a.href = url;

    a.download = `${result.fileName.replace(/\.[^/.]+$/, '')}_extraction.json`;

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);
  };

  /*
   * Helper used for the field cards.
   * We deliberately show "Detected" / "Not detected"
   * instead of fabricated confidence percentages.
   */
  const renderFieldStatus = (value: unknown) => {
    const detected =
      value !== null &&
      value !== undefined &&
      value !== '';

    return detected ? (
      <span className="text-emerald-600 font-mono text-[10px] flex items-center gap-1">
        <CheckCircle2 className="w-3 h-3" />
        Detected
      </span>
    ) : (
      <span className="text-slate-400 font-mono text-[10px] flex items-center gap-1">
        <AlertTriangle className="w-3 h-3" />
        Not detected
      </span>
    );
  };

  /*
   * Currency helper.
   * GSTIN beginning with 27 indicates Maharashtra GST registration.
   * Otherwise we keep the existing UI behaviour.
   */
  const currencySymbol = data.gstin?.startsWith('27') ? '₹ ' : '$ ';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">

      {/* =========================================================
          TOP HEADER
      ========================================================= */}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">

        <div className="flex items-center gap-3">

          <button
            onClick={() => onNavigate('dashboard')}
            className="p-2 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-600 transition-colors"
            title="Back to Dashboard"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div>

            <div className="flex items-center gap-2">

              <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold">
                Extraction Complete
              </span>

              <span className="text-xs text-slate-400 font-mono flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                {result.processingTimeSec}s latency
              </span>

            </div>

            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2 mt-1">
              <span>{result.fileName}</span>
            </h1>

          </div>

        </div>

        {/* Global actions */}

        <div className="flex flex-wrap items-center gap-2.5">

          <button
            id="results-copy-json-btn"
            onClick={handleCopyJson}
            className="px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs border border-slate-200 shadow-xs transition-all flex items-center gap-1.5"
          >

            {copiedJson ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>Copy JSON</span>
              </>
            )}

          </button>

          <button
            id="results-download-json-btn"
            onClick={handleDownloadJson}
            className="px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs border border-slate-200 shadow-xs transition-all flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Download JSON</span>
          </button>

          <button
            id="results-analyze-another-btn"
            onClick={onAnalyzeAnother}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-sm shadow-blue-500/20 transition-all flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Analyze Another</span>
          </button>

        </div>

      </div>


      {/* =========================================================
          EXTRACTION STATUS BAR
      ========================================================= */}

      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white rounded-2xl p-4 sm:p-5 shadow-sm border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">

        <div className="flex items-center gap-4">

          {/* Factual detection count */}

          <div className="w-16 h-12 rounded-xl bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold text-sm font-mono">
            {detectedFieldsCount}/8
          </div>

          <div>

            <div className="flex items-center gap-2 flex-wrap">

              <h3 className="text-sm font-bold text-white">
                Fields Detected
              </h3>

              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-900/60 text-cyan-300 font-bold border border-cyan-700/50">
                {detectedFieldsCount} / 8
              </span>

            </div>

            <p className="text-xs text-slate-300 mt-0.5">
              Values shown below are the fields returned by the Qwen2-VL extraction pipeline.
            </p>

          </div>

        </div>


        {/* Quick field status */}

        <div className="flex flex-wrap items-center gap-2">

          {confidences.slice(0, 5).map((f) => (

            <span
              key={f.field}
              className={`px-2 py-1 rounded-md text-[11px] font-medium flex items-center gap-1 border ${
                f.status === 'valid'
                  ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800'
                  : 'bg-rose-950/60 text-rose-300 border-rose-800'
              }`}
            >

              {f.status === 'valid' ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-3 h-3 text-rose-400" />
              )}

              <span>{f.label}</span>

            </span>

          ))}

        </div>

      </div>


      {/* =========================================================
          MAIN SIDE-BY-SIDE VIEW
      ========================================================= */}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">


        {/* =====================================================
            LEFT COLUMN — ORIGINAL DOCUMENT
        ===================================================== */}

        <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">

          {/* Preview toolbar */}

          <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs">

            <div className="flex items-center gap-2 font-bold text-slate-800">

              <FileText className="w-4 h-4 text-blue-600" />

              <span>Original Document</span>

            </div>

            <div className="flex items-center gap-2">

              <button
                onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                className={`px-2.5 py-1 rounded-lg font-medium flex items-center gap-1 transition-colors ${
                  showBoundingBoxes
                    ? 'bg-blue-100 text-blue-800 border border-blue-200'
                    : 'bg-white text-slate-600 border border-slate-200'
                }`}
                title="Toggle extraction field boxes"
              >

                {showBoundingBoxes ? (
                  <Eye className="w-3.5 h-3.5" />
                ) : (
                  <EyeOff className="w-3.5 h-3.5" />
                )}

                <span>Bounding Boxes</span>

              </button>


              <div className="flex items-center bg-white border border-slate-200 rounded-lg p-0.5">

                <button
                  onClick={() =>
                    setZoomLevel(Math.max(0.7, zoomLevel - 0.15))
                  }
                  className="p-1 hover:bg-slate-100 rounded text-slate-600"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>

                <span className="px-1.5 text-[11px] font-mono text-slate-600">
                  {Math.round(zoomLevel * 100)}%
                </span>

                <button
                  onClick={() =>
                    setZoomLevel(Math.min(1.8, zoomLevel + 0.15))
                  }
                  className="p-1 hover:bg-slate-100 rounded text-slate-600"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>

              </div>

            </div>

          </div>


          {/* Document display */}

          <div className="p-4 bg-slate-100/70 overflow-auto max-h-[640px] flex items-center justify-center relative min-h-[460px]">

            <div
              className="relative shadow-md rounded-lg overflow-hidden transition-transform duration-150 bg-white"
              style={{
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'top center',
              }}
            >

              <img
                src={result.fileUrl}
                alt="Document Preview"
                className="max-w-[560px] w-full h-auto object-contain select-none block"
              />


              {/* Bounding boxes */}

              {showBoundingBoxes &&
                confidences
                  .filter((c) => c.bbox && c.status === 'valid')
                  .map((conf) => {

                    const [ymin, xmin, ymax, xmax] = conf.bbox!;

                    const isHovered =
                      activeHoverField === conf.field;

                    return (

                      <div
                        key={conf.field}
                        onMouseEnter={() =>
                          setActiveHoverField(conf.field)
                        }
                        onMouseLeave={() =>
                          setActiveHoverField(null)
                        }
                        className={`absolute border transition-all cursor-pointer group ${
                          isHovered
                            ? 'border-blue-600 bg-blue-500/25 ring-2 ring-blue-400 z-30'
                            : 'border-emerald-500/80 bg-emerald-500/10 hover:bg-emerald-500/25 z-10'
                        }`}
                        style={{
                          top: `${ymin}%`,
                          left: `${xmin}%`,
                          width: `${xmax - xmin}%`,
                          height: `${ymax - ymin}%`,
                        }}
                      >

                        <div
                          className={`absolute -top-5 left-0 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold whitespace-nowrap shadow-xs pointer-events-none transition-opacity ${
                            isHovered
                              ? 'bg-blue-700 text-white opacity-100 z-40'
                              : 'bg-emerald-700 text-white opacity-0 group-hover:opacity-100'
                          }`}
                        >
                          {conf.label}
                        </div>

                      </div>

                    );

                  })}

            </div>

          </div>


          <div className="p-3 bg-white border-t border-slate-200 text-xs text-slate-500 flex items-center justify-between">

            <span className="font-mono text-[11px]">
              {result.fileType} • {result.fileSize}
            </span>

            <span className="text-[11px] text-slate-400">
              Hover over document boxes to highlight extracted fields
            </span>

          </div>

        </div>


        {/* =====================================================
            RIGHT COLUMN — EXTRACTED INFORMATION
        ===================================================== */}

        <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">


          {/* Tabs */}

          <div className="p-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">

            <div className="flex items-center gap-1 bg-white p-1 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600">

              <button
                onClick={() => setActiveTab('fields')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  activeTab === 'fields'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'hover:text-slate-900 hover:bg-slate-50'
                }`}
              >

                <FileText className="w-3.5 h-3.5" />

                <span>Extracted Fields</span>

              </button>


              <button
                onClick={() => setActiveTab('line_items')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  activeTab === 'line_items'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'hover:text-slate-900 hover:bg-slate-50'
                }`}
              >

                <Table className="w-3.5 h-3.5" />

                <span>
                  Line Items ({data.line_items.length})
                </span>

              </button>


              <button
                onClick={() => setActiveTab('json')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  activeTab === 'json'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'hover:text-slate-900 hover:bg-slate-50'
                }`}
              >

                <Code2 className="w-3.5 h-3.5" />

                <span>Raw JSON</span>

              </button>

            </div>


            <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
              Model: {result.modelName.split(' ')[0]}
            </span>

          </div>


          {/* =================================================
              TAB 1 — EXTRACTED FIELDS
          ================================================= */}

          {activeTab === 'fields' && (

            <div className="p-5 space-y-4 max-h-[640px] overflow-y-auto">

              <div className="flex items-center justify-between">

                <div>

                  <h3 className="text-sm font-bold text-slate-900">
                    Document Key-Value Pairs
                  </h3>

                  <p className="text-xs text-slate-500">
                    Fields extracted from the uploaded document
                  </p>

                </div>

                <span className="text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-bold border border-emerald-200">
                  {detectedFieldsCount} / 8 detected
                </span>

              </div>


              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">


                {/* Vendor Name */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('vendor_name')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'vendor_name'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>Vendor Name</span>

                    {renderFieldStatus(data.vendor_name)}

                  </div>

                  <div className="text-sm font-bold text-slate-900">

                    {data.vendor_name || (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* Invoice Number */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('invoice_number')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'invoice_number'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>Invoice Number</span>

                    {renderFieldStatus(data.invoice_number)}

                  </div>

                  <div className="text-sm font-bold text-slate-900 font-mono">

                    {data.invoice_number ? (
                      `#${data.invoice_number}`
                    ) : (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* Invoice Date */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('invoice_date')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'invoice_date'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>Invoice Date</span>

                    {renderFieldStatus(data.invoice_date)}

                  </div>

                  <div className="text-sm font-semibold text-slate-800">

                    {data.invoice_date || (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* GSTIN */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('gstin')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'gstin'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>GSTIN / Tax ID</span>

                    {renderFieldStatus(data.gstin)}

                  </div>

                  <div className="text-sm font-bold text-blue-700 font-mono">

                    {data.gstin || (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* State */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('state')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'state'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>State</span>

                    {renderFieldStatus(data.state)}

                  </div>

                  <div className="text-sm font-semibold text-slate-800">

                    {data.state || (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* State Code */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('state_code')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'state_code'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>State Code</span>

                    {renderFieldStatus(data.state_code)}

                  </div>

                  <div className="text-sm font-bold text-slate-800 font-mono">

                    {data.state_code || (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* Total Tax */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('total_tax')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all ${
                    activeHoverField === 'total_tax'
                      ? 'border-blue-500 bg-blue-50/50 shadow-xs'
                      : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">

                    <span>Total Tax</span>

                    {renderFieldStatus(data.total_tax)}

                  </div>

                  <div className="text-sm font-bold text-slate-800">

                    {data.total_tax !== null ? (
                      `${currencySymbol}${data.total_tax.toLocaleString()}`
                    ) : (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>


                {/* Total Amount */}

                <div
                  onMouseEnter={() =>
                    setActiveHoverField('total_amount')
                  }
                  onMouseLeave={() =>
                    setActiveHoverField(null)
                  }
                  className={`p-3.5 rounded-xl border transition-all bg-gradient-to-br from-blue-50/50 to-indigo-50/30 ${
                    activeHoverField === 'total_amount'
                      ? 'border-blue-600 ring-2 ring-blue-300'
                      : 'border-blue-200 hover:border-blue-300'
                  }`}
                >

                  <div className="flex items-center justify-between text-[11px] font-semibold text-blue-700 mb-1">

                    <span>Total Amount</span>

                    {renderFieldStatus(data.total_amount)}

                  </div>

                  <div className="text-base font-extrabold text-blue-900">

                    {data.total_amount !== null ? (
                      `${currencySymbol}${data.total_amount.toLocaleString()}`
                    ) : (
                      <span className="text-slate-400 italic">
                        Not detected
                      </span>
                    )}

                  </div>

                </div>

              </div>


              {/* Line item preview */}

              {data.line_items.length > 0 && (

                <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">

                  <div className="text-xs">

                    <span className="font-bold text-slate-800">
                      {data.line_items.length} Line Items Extracted
                    </span>

                    <p className="text-[11px] text-slate-500">
                      Structured line items returned by the extraction model.
                    </p>

                  </div>

                  <button
                    onClick={() => setActiveTab('line_items')}
                    className="px-3 py-1.5 rounded-lg bg-blue-600 text-white font-semibold text-xs hover:bg-blue-700"
                  >
                    View Line Items Table →
                  </button>

                </div>

              )}

            </div>

          )}


          {/* =================================================
              TAB 2 — LINE ITEMS
          ================================================= */}

          {activeTab === 'line_items' && (

            <div className="p-5 space-y-4 max-h-[640px] overflow-y-auto">

              <div className="flex items-center justify-between">

                <div>

                  <h3 className="text-sm font-bold text-slate-900">
                    Extracted Line Items
                  </h3>

                  <p className="text-xs text-slate-500">
                    Structured line items returned by Qwen2-VL.
                  </p>

                </div>

                <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2.5 py-1 rounded-md font-semibold">
                  {data.line_items.length} Items Found
                </span>

              </div>


              {data.line_items.length === 0 ? (

                <div className="p-10 text-center text-slate-400 text-xs bg-slate-50 rounded-xl border border-dashed border-slate-200">
                  No line items detected for this document.
                </div>

              ) : (

                <div className="border border-slate-200 rounded-xl overflow-hidden shadow-xs">

                  <table className="w-full text-left text-xs border-collapse">

                    <thead>

                      <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[10px]">

                        <th className="py-2.5 px-3 w-10 text-center">
                          #
                        </th>

                        <th className="py-2.5 px-3">
                          Item Description
                        </th>

                        <th className="py-2.5 px-3 text-center w-16">
                          Quantity
                        </th>

                        <th className="py-2.5 px-3 text-right w-24">
                          Price
                        </th>

                        <th className="py-2.5 px-3 text-right w-28">
                          Total
                        </th>

                      </tr>

                    </thead>


                    <tbody className="divide-y divide-slate-100">

                      {data.line_items.map((item, idx) => (

                        <tr
                          key={item.id || idx}
                          className="hover:bg-slate-50/80 transition-colors"
                        >

                          <td className="py-3 px-3 text-center text-slate-400 font-mono">
                            {idx + 1}
                          </td>

                          <td className="py-3 px-3 font-semibold text-slate-800">
                            {item.item}
                          </td>

                          <td className="py-3 px-3 text-center text-slate-700 font-mono">
                            {item.quantity}
                          </td>

                          <td className="py-3 px-3 text-right text-slate-600 font-mono">

                            {item.unit_price !== undefined
                              ? `${currencySymbol}${item.unit_price.toLocaleString()}`
                              : '-'}

                          </td>

                          <td className="py-3 px-3 text-right font-bold text-slate-900 font-mono">

                            {currencySymbol}

                            {item.price.toLocaleString()}

                          </td>

                        </tr>

                      ))}

                    </tbody>


                    <tfoot className="bg-slate-50 border-t border-slate-200 font-bold text-slate-900">

                      <tr>

                        <td
                          colSpan={4}
                          className="py-2.5 px-3 text-right text-slate-500"
                        >
                          Total Invoice Value:
                        </td>

                        <td className="py-2.5 px-3 text-right text-blue-700 font-mono text-sm">

                          {data.total_amount !== null
                            ? `${currencySymbol}${data.total_amount.toLocaleString()}`
                            : '-'}

                        </td>

                      </tr>

                    </tfoot>

                  </table>

                </div>

              )}


              <div className="p-3 bg-blue-50/60 rounded-xl border border-blue-200/80 text-xs text-blue-900 flex items-center justify-between">

                <span>
                  Structured according to the project's invoice extraction schema.
                </span>

                <button
                  onClick={() => setActiveTab('json')}
                  className="font-bold underline hover:text-blue-700"
                >
                  View JSON Schema
                </button>

              </div>

            </div>

          )}


          {/* =================================================
              TAB 3 — RAW JSON
          ================================================= */}

          {activeTab === 'json' && (

            <div className="p-5 space-y-4 max-h-[640px] overflow-y-auto">

              <div className="flex items-center justify-between">

                <div>

                  <h3 className="text-sm font-bold text-slate-900">
                    Structured JSON Output
                  </h3>

                  <p className="text-xs text-slate-500">
                    JSON returned from the invoice extraction pipeline.
                  </p>

                </div>


                <div className="flex items-center gap-2">

                  <button
                    onClick={handleCopyJson}
                    className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-semibold text-slate-700 flex items-center gap-1"
                  >

                    {copiedJson ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}

                    <span>
                      {copiedJson ? 'Copied' : 'Copy'}
                    </span>

                  </button>


                  <button
                    onClick={handleDownloadJson}
                    className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-xs font-semibold text-white flex items-center gap-1"
                  >

                    <Download className="w-3.5 h-3.5" />

                    <span>Download</span>

                  </button>

                </div>

              </div>


              {/* JSON viewer */}

              <div className="bg-slate-900 text-slate-100 rounded-xl p-4 font-mono text-xs overflow-x-auto shadow-inner border border-slate-800">

                <pre className="leading-relaxed text-emerald-400">
                  {formattedJsonString}
                </pre>

              </div>


              <div className="text-[11px] text-slate-400 p-2 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between">

                <span>
                  Contract: POST /api/analyze return format
                </span>

                <span className="font-mono text-slate-600">
                  application/json
                </span>

              </div>

            </div>

          )}

        </div>

      </div>

    </div>
  );
};