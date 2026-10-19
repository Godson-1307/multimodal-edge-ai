import { ExtractionData, ExtractionResult, FieldConfidence } from '../types';

export interface AnalyzeOptions {
  apiUrl?: string;
  onStageChange?: (
    stageIndex: number,
    stageName: string
  ) => void;
  simulateDelay?: number;
}

/**
 * Normalizes the real FastAPI backend response
 * into the frontend ExtractionResult contract.
 */
export function normalizeBackendResponse(
  rawJson: Partial<ExtractionData> & Record<string, any>,
  fileMeta: { name: string; size: string; type: string; previewUrl: string },
  processingTimeSec: number
): ExtractionResult {
  const data: ExtractionData = {
    vendor_name: rawJson.vendor_name ?? rawJson.vendor ?? rawJson.company_name ?? null,

    invoice_number:
      rawJson.invoice_number ??
      rawJson.invoice_no ??
      rawJson.inv_no ??
      null,

    invoice_date:
      rawJson.invoice_date ??
      rawJson.date ??
      null,

    gstin:
      rawJson.gstin ??
      rawJson.tax_id ??
      rawJson.gst_number ??
      null,

    state: rawJson.state ?? null,

    state_code: rawJson.state_code ?? null,

    total_tax:
      typeof rawJson.total_tax === 'number'
        ? rawJson.total_tax
        : rawJson.total_tax
          ? parseFloat(rawJson.total_tax)
          : null,

    total_amount:
      typeof rawJson.total_amount === 'number'
        ? rawJson.total_amount
        : rawJson.total_amount
          ? parseFloat(rawJson.total_amount)
          : null,

    line_items: Array.isArray(rawJson.line_items)
      ? rawJson.line_items.map((item: any, idx: number) => ({
          id: String(idx + 1),

          item:
            item.item ??
            item.name ??
            item.description ??
            'Item',

          quantity: Number(
            item.quantity ??
            item.qty ??
            1
          ),

          unit_price:
            item.unit_price !== undefined &&
            item.unit_price !== null
              ? Number(item.unit_price)
              : undefined,

          price: Number(
            item.price ??
            item.amount ??
            item.total ??
            0
          ),
        }))
      : [],
  };

  /*
   * Frontend display confidence.
   *
   * IMPORTANT:
   * These are UI confidence indicators based on whether
   * the backend returned a value. They are NOT model-calculated
   * confidence scores.
   */
  const fieldsToCheck: {
    key: keyof ExtractionData;
    label: string;
    defaultBbox?: [number, number, number, number];
  }[] = [
    {
      key: 'vendor_name',
      label: 'Vendor Name',
      defaultBbox: [5, 6, 11, 45],
    },
    {
      key: 'invoice_number',
      label: 'Invoice Number',
      defaultBbox: [5, 68, 9, 88],
    },
    {
      key: 'invoice_date',
      label: 'Invoice Date',
      defaultBbox: [9, 68, 12, 88],
    },
    {
      key: 'gstin',
      label: 'GSTIN',
      defaultBbox: [12, 6, 15, 35],
    },
    {
      key: 'state',
      label: 'State',
      defaultBbox: [18, 6, 21, 28],
    },
    {
      key: 'state_code',
      label: 'State Code',
      defaultBbox: [18, 30, 21, 38],
    },
    {
      key: 'total_tax',
      label: 'Total Tax',
      defaultBbox: [62, 55, 67, 88],
    },
    {
      key: 'total_amount',
      label: 'Total Amount',
      defaultBbox: [68, 55, 74, 88],
    },
  ];

  let detectedCount = 0;

  const confidences: FieldConfidence[] = fieldsToCheck.map((f) => {
    const val = data[f.key];

    const isDetected =
      val !== null &&
      val !== undefined &&
      val !== '';

    if (isDetected) {
      detectedCount++;
    }

    /*
     * These values are only placeholders for the existing UI.
     * They should NOT be interpreted as actual Qwen confidence.
     */
    const confidence = isDetected ? 95 : 15;

    return {
      field: f.key,
      label: f.label,

      value: isDetected
        ? typeof val === 'object'
          ? JSON.stringify(val)
          : String(val)
        : null,

      confidence,

      status: isDetected
        ? 'valid'
        : 'not_detected',

      bbox: isDetected
        ? f.defaultBbox
        : undefined,
    };
  });

  const qualityScore = Math.round(
    (detectedCount / fieldsToCheck.length) * 98
  );

  return {
    id: `doc-${Date.now()}`,

    fileName: fileMeta.name,

    fileSize: fileMeta.size,

    fileType: fileMeta.type,

    fileUrl: fileMeta.previewUrl,

    uploadedAt: new Date().toLocaleString(
      'en-GB',
      {
        dateStyle: 'short',
        timeStyle: 'short',
      }
    ),

    status: 'Completed',

    processingTimeSec:
      Math.round(processingTimeSec * 100) / 100,

    modelName:
      'Qwen2-VL-2B (QLoRA Fine-Tuned)',

    data,

    confidences,

    qualityScore,

    extractedFieldsCount: detectedCount,

    totalFieldsCount: fieldsToCheck.length,
  };
}


/**
 * Sends the uploaded document to the real FastAPI backend.
 */
export async function analyzeDocument(
  file: File | {
    name: string;
    size: number;
    type: string;
    dataUrl?: string;
  },
  options: AnalyzeOptions = {}
): Promise<ExtractionResult> {

  const {
    apiUrl,
    onStageChange,
  } = options;

  const defaultUrl =
    (import.meta as any).env?.VITE_API_URL ||
    'http://localhost:8000/api/analyze';

  const targetApiUrl =
    apiUrl || defaultUrl;


  const fileSizeString =
    file.size > 1024 * 1024
      ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
      : `${Math.round(file.size / 1024)} KB`;


  /*
   * Create browser preview URL.
   */
  let previewUrl =
    (file as any).dataUrl || '';

  if (!previewUrl && file instanceof File) {
    previewUrl =
      URL.createObjectURL(file);
  }


  const fileMeta = {
    name: file.name,

    size: fileSizeString,

    type:
      file.type ||
      'image/jpeg',

    previewUrl,
  };


  const startTime =
    performance.now();


  try {

    onStageChange?.(
      0,
      'Initializing Vision Pipeline'
    );


    /*
     * Create multipart/form-data request.
     */
    const formData =
      new FormData();


    if (file instanceof File) {

      /*
       * FastAPI expects:
       * document: UploadFile
       */
      formData.append(
        'document',
        file
      );

    } else {

      throw new Error(
        'Invalid file object. Please upload the document again.'
      );

    }


    /*
     * Qwen inference can take several seconds,
     * so allow up to 120 seconds.
     */
    const controller =
      new AbortController();

    const timeoutId =
      setTimeout(
        () => controller.abort(),
        120000
      );


    onStageChange?.(
      1,
      'Reading document & Vision Tokenization'
    );


    onStageChange?.(
      2,
      'Running Qwen2-VL + QLoRA inference'
    );


    const res =
      await fetch(
        targetApiUrl,
        {
          method: 'POST',

          body: formData,

          signal:
            controller.signal,
        }
      );


    clearTimeout(timeoutId);


    /*
     * Backend returned an HTTP error.
     */
    if (!res.ok) {

      let errorMessage =
        `Backend returned HTTP ${res.status}`;

      try {

        const errorJson =
          await res.json();

        if (errorJson?.detail) {
          errorMessage =
            errorJson.detail;
        }

      } catch {
        // Keep default error message.
      }

      throw new Error(
        errorMessage
      );
    }


    onStageChange?.(
      3,
      'Receiving Model Inferences'
    );


    const json =
      await res.json();


    const elapsedSec =
      (performance.now() -
        startTime) / 1000;


    onStageChange?.(
      4,
      'Formatting Results'
    );


    return normalizeBackendResponse(
      json,
      fileMeta,
      elapsedSec
    );


  } catch (err: any) {

    /*
     * Convert timeout into a useful message.
     */
    if (
      err?.name ===
      'AbortError'
    ) {

      throw new Error(
        'The OCR model took too long to respond. Please try again.'
      );

    }


    /*
     * Network / backend error.
     */
    if (
      err instanceof TypeError
    ) {

      throw new Error(
        'Cannot connect to the OCR backend. Make sure FastAPI is running on http://localhost:8000.'
      );

    }


    throw err;
  }
}


/**
 * Ping the backend health endpoint.
 */
export async function pingBackend(
  url: string
): Promise<{
  online: boolean;
  latencyMs?: number;
  message: string;
}> {

  const start =
    performance.now();


  try {

    const res =
      await fetch(
        url,
        {
          method: 'GET',
          mode: 'cors',
        }
      );


    const latencyMs =
      Math.round(
        performance.now() -
        start
      );


    return {

      online:
        res.ok,

      latencyMs,

      message:
        res.ok
          ? `Connected (${latencyMs}ms)`
          : `Status code ${res.status}`,
    };


  } catch {

    return {

      online: false,

      message:
        'Unreachable (Edge API server offline or CORS disabled)',
    };
  }
}
