export type AppView =
  | 'landing'
  | 'login'
  | 'register'
  | 'dashboard'
  | 'upload'
  | 'processing'
  | 'results'
  | 'history'
  | 'details'
  | 'profile';

export interface LineItem {
  id?: string;
  item: string;
  quantity: number;
  unit_price?: number;
  price: number;
}

export interface FieldConfidence {
  field: string;
  label: string;
  value: string | number | null;
  confidence: number; // 0 - 100
  status: 'valid' | 'warning' | 'not_detected';
  bbox?: [number, number, number, number]; // [ymin, xmin, ymax, xmax] in percentages (0-100)
}

export interface ExtractionData {
  vendor_name: string | null;
  invoice_number: string | null;
  invoice_date: string | null;
  gstin: string | null;
  state: string | null;
  state_code: string | null;
  total_tax: number | null;
  total_amount: number | null;
  line_items: LineItem[];
}

export interface ExtractionResult {
  id: string;
  fileName: string;
  fileSize: string;
  fileType: string;
  fileUrl: string;
  uploadedAt: string;
  status: 'Completed' | 'Processing' | 'Failed';
  processingTimeSec: number;
  modelName: string; // e.g. "Qwen2-VL-2B (QLoRA Fine-tuned)"
  data: ExtractionData;
  confidences: FieldConfidence[];
  qualityScore: number; // 0 - 100
  extractedFieldsCount: number;
  totalFieldsCount: number;
}

export interface ProcessingStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  durationMs?: number;
}

export interface UserProfile {
  name: string;
  email: string;
  role: string;
  avatarUrl?: string;
}

export interface EdgeSettings {
  darkMode: boolean;
  saveHistory: boolean;
  backendApiUrl: string;
  useCustomBackend: boolean;
  modelType: 'qwen2-vl-2b' | 'qwen2-vl-7b';
  deviceTarget: 'edge-gpu' | 'cpu' | 'npu';
  simulateDelay: number; // ms
}
