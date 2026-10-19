import { ExtractionResult } from './types';

// Helper to generate realistic SVG data URI for document preview
export function generateInvoiceSvg(params: {
  title: string;
  vendor: string;
  invNumber: string;
  date: string;
  gstin?: string;
  address?: string;
  items: { name: string; qty: number; rate: number; amount: number }[];
  subtotal: number;
  tax: number;
  total: number;
  currency?: string;
  highlightBoxes?: boolean;
}): string {
  const { title, vendor, invNumber, date, gstin, address, items, subtotal, tax, total, currency = '₹' } = params;

  const itemsRows = items
    .map(
      (it, idx) => `
    <tr style="border-bottom: 1px solid #e2e8f0; font-size: 13px;">
      <td style="padding: 10px 8px; color: #334155;">${idx + 1}</td>
      <td style="padding: 10px 8px; font-weight: 500; color: #0f172a;">${it.name}</td>
      <td style="padding: 10px 8px; text-align: center; color: #334155;">${it.qty}</td>
      <td style="padding: 10px 8px; text-align: right; color: #334155;">${currency} ${it.rate.toLocaleString()}</td>
      <td style="padding: 10px 8px; text-align: right; font-weight: 600; color: #0f172a;">${currency} ${it.amount.toLocaleString()}</td>
    </tr>`
    )
    .join('');

  const svgContent = `
  <svg xmlns="http://www.w3.org/2000/svg" width="680" height="900" viewBox="0 0 680 900" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <defs>
      <linearGradient id="headerGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#0f172a"/>
        <stop offset="100%" stop-color="#1e293b"/>
      </linearGradient>
      <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f8fafc" stroke-width="1"/>
      </pattern>
    </defs>

    <!-- Background paper -->
    <rect width="680" height="900" fill="#fcfdfd"/>
    <rect width="680" height="900" fill="url(#grid)" opacity="0.6"/>

    <!-- Subtle border & paper fold shadow -->
    <rect x="15" y="15" width="650" height="870" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.03))"/>

    <!-- Header bar -->
    <rect x="15" y="15" width="650" height="12" rx="4" fill="url(#headerGrad)"/>

    <g transform="translate(45, 55)">
      <!-- Company Branding -->
      <text x="0" y="24" font-size="22" font-weight="800" fill="#0f172a" letter-spacing="-0.5">${vendor}</text>
      <text x="0" y="44" font-size="12" fill="#64748b">${address || 'Industrial Area Phase 2, Mumbai, MH, India'}</text>
      ${gstin ? `<text x="0" y="62" font-size="12" font-weight="600" fill="#334155">GSTIN: <tspan font-family="monospace" fill="#0284c7">${gstin}</tspan></text>` : ''}

      <!-- Document Meta Badge -->
      <g transform="translate(430, 0)">
        <rect x="0" y="0" width="160" height="34" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
        <text x="80" y="22" font-size="14" font-weight="700" fill="#0f172a" text-anchor="middle" letter-spacing="1">${title.toUpperCase()}</text>
        
        <text x="160" y="55" font-size="12" fill="#64748b" text-anchor="end">Invoice No: <tspan font-weight="700" fill="#0f172a">#${invNumber}</tspan></text>
        <text x="160" y="73" font-size="12" fill="#64748b" text-anchor="end">Date: <tspan font-weight="600" fill="#0f172a">${date}</tspan></text>
      </g>

      <!-- Divider line -->
      <line x1="0" y1="95" x2="590" y2="95" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="4 4"/>

      <!-- Bill to info -->
      <g transform="translate(0, 115)">
        <text x="0" y="0" font-size="11" font-weight="700" fill="#94a3b8" letter-spacing="0.5">BILLED TO:</text>
        <text x="0" y="18" font-size="14" font-weight="700" fill="#1e293b">Retail Consumer & Distribution Co.</text>
        <text x="0" y="34" font-size="12" fill="#64748b">Sector 14, Commercial Hub</text>
        <text x="0" y="50" font-size="12" fill="#64748b">State: Maharashtra (Code: 27)</text>
      </g>

      <!-- Line items table embedded via foreignObject -->
      <foreignObject x="0" y="195" width="590" height="380">
        <div xmlns="http://www.w3.org/1999/xhtml">
          <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
              <tr style="background: #f8fafc; border-top: 1px solid #cbd5e1; border-bottom: 2px solid #cbd5e1; font-size: 11px; font-weight: 700; color: #475569; letter-spacing: 0.5px; text-transform: uppercase;">
                <th style="padding: 10px 8px; width: 30px;">#</th>
                <th style="padding: 10px 8px;">Item Description</th>
                <th style="padding: 10px 8px; text-align: center; width: 60px;">Qty</th>
                <th style="padding: 10px 8px; text-align: right; width: 110px;">Price</th>
                <th style="padding: 10px 8px; text-align: right; width: 120px;">Amount</th>
              </tr>
            </thead>
            <tbody>
              ${itemsRows}
            </tbody>
          </table>
        </div>
      </foreignObject>

      <!-- Totals summary block -->
      <g transform="translate(340, 560)">
        <rect x="0" y="0" width="250" height="130" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
        <text x="20" y="32" font-size="13" fill="#64748b">Sub Total:</text>
        <text x="230" y="32" font-size="13" font-weight="600" fill="#0f172a" text-anchor="end">${currency} ${subtotal.toLocaleString()}</text>
        
        <text x="20" y="60" font-size="13" fill="#64748b">Tax / GST:</text>
        <text x="230" y="60" font-size="13" font-weight="600" fill="#0284c7" text-anchor="end">${currency} ${tax.toLocaleString()}</text>

        <line x1="15" y1="78" x2="235" y2="78" stroke="#cbd5e1" stroke-width="1"/>

        <text x="20" y="105" font-size="15" font-weight="800" fill="#0f172a">Total Amount:</text>
        <text x="230" y="105" font-size="16" font-weight="800" fill="#0f172a" text-anchor="end">${currency} ${total.toLocaleString()}</text>
      </g>

      <!-- Verification stamp -->
      <g transform="translate(40, 600) rotate(-6)">
        <rect x="0" y="0" width="160" height="60" rx="6" fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="6 3"/>
        <text x="80" y="26" font-size="12" font-weight="800" fill="#059669" text-anchor="middle" letter-spacing="1">EDGE AI PROCESSED</text>
        <text x="80" y="44" font-size="10" font-weight="600" fill="#059669" text-anchor="middle">QWEN2-VL OCR CONFIRMED</text>
      </g>

      <!-- Footer note -->
      <g transform="translate(0, 750)">
        <line x1="0" y1="0" x2="590" y2="0" stroke="#f1f5f9" stroke-width="1"/>
        <text x="295" y="24" font-size="11" fill="#94a3b8" text-anchor="middle">Computer Generated Invoice • Processed with Multi-Modal Edge AI QLoRA</text>
      </g>
    </g>
  </svg>
  `;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svgContent)}`;
}

// 1. Exact "TOP STAR NAMKEEN" invoice from the user's prompt
export const MOCK_INVOICE_TOPSTAR: ExtractionResult = {
  id: 'doc-topstar-725',
  fileName: 'invoice_topstar_725.jpg',
  fileSize: '2.4 MB',
  fileType: 'image/jpeg',
  fileUrl: generateInvoiceSvg({
    title: 'Tax Invoice',
    vendor: 'TOP STAR NAMKEEN',
    invNumber: '725',
    date: '11/07/2026',
    gstin: '27AGFPN1247B1ZS',
    address: 'Plot No. 44, MIDC Industrial Area, Pune, Maharashtra 411019',
    items: [
      { name: 'Special Ratlami Sev (500g Jar)', qty: 450, rate: 1200, amount: 540000 },
      { name: 'Bikaneri Bhujia Premium Pack', qty: 320, rate: 1500, amount: 480000 },
      { name: 'Roasted Salted Cashews & Almond Mix', qty: 150, rate: 2846, amount: 426905 },
    ],
    subtotal: 1446905,
    tax: 144695,
    total: 1591600,
    currency: '₹',
  }),
  uploadedAt: '11/07/2026, 14:22',
  status: 'Completed',
  processingTimeSec: 2.14,
  modelName: 'Qwen2-VL-2B (QLoRA Fine-Tuned)',
  data: {
    vendor_name: 'TOP STAR NAMKEEN',
    invoice_number: '725',
    invoice_date: '11/07/2026',
    gstin: '27AGFPN1247B1ZS',
    state: 'Maharashtra',
    state_code: '27',
    total_tax: 144695,
    total_amount: 1591600,
    line_items: [
      { id: '1', item: 'Special Ratlami Sev (500g Jar)', quantity: 450, unit_price: 1200, price: 540000 },
      { id: '2', item: 'Bikaneri Bhujia Premium Pack', quantity: 320, unit_price: 1500, price: 480000 },
      { id: '3', item: 'Roasted Salted Cashews & Almond Mix', quantity: 150, unit_price: 2846, price: 426905 },
    ],
  },
  confidences: [
    { field: 'vendor_name', label: 'Vendor Name', value: 'TOP STAR NAMKEEN', confidence: 99.4, status: 'valid', bbox: [5, 6, 11, 45] },
    { field: 'invoice_number', label: 'Invoice Number', value: '725', confidence: 99.8, status: 'valid', bbox: [5, 68, 9, 88] },
    { field: 'invoice_date', label: 'Invoice Date', value: '11/07/2026', confidence: 98.6, status: 'valid', bbox: [9, 68, 12, 88] },
    { field: 'gstin', label: 'GSTIN', value: '27AGFPN1247B1ZS', confidence: 99.1, status: 'valid', bbox: [12, 6, 15, 35] },
    { field: 'state', label: 'State', value: 'Maharashtra', confidence: 97.2, status: 'valid', bbox: [18, 6, 21, 28] },
    { field: 'state_code', label: 'State Code', value: '27', confidence: 98.9, status: 'valid', bbox: [18, 30, 21, 38] },
    { field: 'total_tax', label: 'Total Tax', value: '₹ 1,44,695', confidence: 96.5, status: 'valid', bbox: [62, 55, 67, 88] },
    { field: 'total_amount', label: 'Total Amount', value: '₹ 15,91,600', confidence: 99.7, status: 'valid', bbox: [68, 55, 74, 88] },
  ],
  qualityScore: 98.7,
  extractedFieldsCount: 8,
  totalFieldsCount: 8,
};

// 2. CORD training dataset receipt (with Nasi Campur Bali, Bbk Bengil Nasi)
export const MOCK_INVOICE_CORD: ExtractionResult = {
  id: 'doc-cord-bali-082',
  fileName: 'receipt_cord_bali.jpg',
  fileSize: '1.8 MB',
  fileType: 'image/jpeg',
  fileUrl: generateInvoiceSvg({
    title: 'Dining Receipt',
    vendor: 'BEBEK BENGIL RESTO & CAFE',
    invNumber: 'ORD-8842',
    date: '08/09/2026',
    gstin: 'ID-0847291-BALI',
    address: 'Jl. Hanoman, Padangtegal, Ubud, Bali 80571',
    items: [
      { name: 'Nasi Campur Bali Special', qty: 1, rate: 75000, amount: 75000 },
      { name: 'Bbk Bengil Nasi Crispy Duck', qty: 2, rate: 120000, amount: 240000 },
      { name: 'Fresh Lime & Lemongrass Tea', qty: 3, rate: 25000, amount: 75000 },
    ],
    subtotal: 390000,
    tax: 39000,
    total: 429000,
    currency: 'IDR',
  }),
  uploadedAt: '08/09/2026, 19:40',
  status: 'Completed',
  processingTimeSec: 1.85,
  modelName: 'Qwen2-VL-2B (CORD Benchmark)',
  data: {
    vendor_name: 'BEBEK BENGIL RESTO & CAFE',
    invoice_number: 'ORD-8842',
    invoice_date: '08/09/2026',
    gstin: 'ID-0847291-BALI',
    state: 'Bali',
    state_code: '51',
    total_tax: 39000,
    total_amount: 429000,
    line_items: [
      { id: '1', item: 'Nasi Campur Bali Special', quantity: 1, unit_price: 75000, price: 75000 },
      { id: '2', item: 'Bbk Bengil Nasi Crispy Duck', quantity: 2, unit_price: 120000, price: 240000 },
      { id: '3', item: 'Fresh Lime & Lemongrass Tea', quantity: 3, unit_price: 25000, price: 75000 },
    ],
  },
  confidences: [
    { field: 'vendor_name', label: 'Vendor Name', value: 'BEBEK BENGIL RESTO & CAFE', confidence: 98.9, status: 'valid', bbox: [5, 6, 11, 55] },
    { field: 'invoice_number', label: 'Invoice Number', value: 'ORD-8842', confidence: 99.2, status: 'valid', bbox: [5, 68, 9, 88] },
    { field: 'invoice_date', label: 'Invoice Date', value: '08/09/2026', confidence: 97.4, status: 'valid', bbox: [9, 68, 12, 88] },
    { field: 'gstin', label: 'Tax ID / NPWP', value: 'ID-0847291-BALI', confidence: 95.0, status: 'valid', bbox: [12, 6, 15, 35] },
    { field: 'state', label: 'State / Province', value: 'Bali', confidence: 96.1, status: 'valid', bbox: [18, 6, 21, 28] },
    { field: 'state_code', label: 'Province Code', value: '51', confidence: 94.8, status: 'valid', bbox: [18, 30, 21, 38] },
    { field: 'total_tax', label: 'Total Tax (PB1)', value: 'IDR 39,000', confidence: 96.0, status: 'valid', bbox: [62, 55, 67, 88] },
    { field: 'total_amount', label: 'Total Amount', value: 'IDR 429,000', confidence: 99.5, status: 'valid', bbox: [68, 55, 74, 88] },
  ],
  qualityScore: 97.1,
  extractedFieldsCount: 8,
  totalFieldsCount: 8,
};

// 3. Vertex Solutions invoice (inspired by the uploaded design mockup!)
export const MOCK_INVOICE_VERTEX: ExtractionResult = {
  id: 'doc-inv-0254',
  fileName: 'INV-0254_VertexSolutions.pdf',
  fileSize: '3.1 MB',
  fileType: 'application/pdf',
  fileUrl: generateInvoiceSvg({
    title: 'Enterprise Billing',
    vendor: 'Vertex Solutions Inc.',
    invNumber: 'INV-0254',
    date: '15/11/2026',
    gstin: 'US-992014-CA',
    address: '450 Innovation Parkway, Suite 700, San Jose, CA',
    items: [
      { name: 'Multi-Modal Edge AI GPU Server Cluster Node', qty: 2, rate: 5000, amount: 10000 },
      { name: 'Qwen2-VL Fine-Tuning & Quantization Pipeline', qty: 1, rate: 2500, amount: 2500 },
    ],
    subtotal: 12500,
    tax: 0,
    total: 12500,
    currency: '$',
  }),
  uploadedAt: '15/11/2026, 10:15',
  status: 'Completed',
  processingTimeSec: 1.42,
  modelName: 'Qwen2-VL-2B (Edge TensorRT-LLM)',
  data: {
    vendor_name: 'Vertex Solutions Inc.',
    invoice_number: 'INV-0254',
    invoice_date: '15/11/2026',
    gstin: 'US-992014-CA',
    state: 'California',
    state_code: '06',
    total_tax: 0,
    total_amount: 12500,
    line_items: [
      { id: '1', item: 'Multi-Modal Edge AI GPU Server Cluster Node', quantity: 2, unit_price: 5000, price: 10000 },
      { id: '2', item: 'Qwen2-VL Fine-Tuning & Quantization Pipeline', quantity: 1, unit_price: 2500, price: 2500 },
    ],
  },
  confidences: [
    { field: 'vendor_name', label: 'Vendor Name', value: 'Vertex Solutions Inc.', confidence: 99.8, status: 'valid', bbox: [5, 6, 11, 45] },
    { field: 'invoice_number', label: 'Invoice Number', value: 'INV-0254', confidence: 99.9, status: 'valid', bbox: [5, 68, 9, 88] },
    { field: 'invoice_date', label: 'Invoice Date', value: '15/11/2026', confidence: 99.0, status: 'valid', bbox: [9, 68, 12, 88] },
    { field: 'gstin', label: 'Tax ID', value: 'US-992014-CA', confidence: 98.4, status: 'valid', bbox: [12, 6, 15, 35] },
    { field: 'state', label: 'State', value: 'California', confidence: 98.1, status: 'valid', bbox: [18, 6, 21, 28] },
    { field: 'state_code', label: 'State Code', value: '06', confidence: 97.5, status: 'valid', bbox: [18, 30, 21, 38] },
    { field: 'total_tax', label: 'Total Tax', value: '$ 0.00', confidence: 95.0, status: 'valid', bbox: [62, 55, 67, 88] },
    { field: 'total_amount', label: 'Total Amount', value: '$ 12,500.00', confidence: 99.9, status: 'valid', bbox: [68, 55, 74, 88] },
  ],
  qualityScore: 98.5,
  extractedFieldsCount: 8,
  totalFieldsCount: 8,
};

// 4. Sample with partial detection warning to demonstrate realistic validation
export const MOCK_INVOICE_WARNING: ExtractionResult = {
  id: 'doc-rec-crumpled-009',
  fileName: 'receipt_crumpled_009.jpg',
  fileSize: '840 KB',
  fileType: 'image/jpeg',
  fileUrl: generateInvoiceSvg({
    title: 'Thermal Cash Receipt',
    vendor: 'Metro Quick Supermarket',
    invNumber: 'CSH-9921',
    date: '04/09/2026',
    address: 'Corner Bay 12, City Center',
    items: [
      { name: 'Organic Milk 1L', qty: 2, rate: 3.5, amount: 7.0 },
      { name: 'Whole Wheat Sourdough Loaf', qty: 1, rate: 4.2, amount: 4.2 },
    ],
    subtotal: 11.2,
    tax: 0.56,
    total: 11.76,
    currency: '$',
  }),
  uploadedAt: '04/09/2026, 11:05',
  status: 'Completed',
  processingTimeSec: 2.89,
  modelName: 'Qwen2-VL-2B (Fine-Tuned)',
  data: {
    vendor_name: 'Metro Quick Supermarket',
    invoice_number: 'CSH-9921',
    invoice_date: '04/09/2026',
    gstin: null,
    state: null,
    state_code: null,
    total_tax: 0.56,
    total_amount: 11.76,
    line_items: [
      { id: '1', item: 'Organic Milk 1L', quantity: 2, unit_price: 3.5, price: 7.0 },
      { id: '2', item: 'Whole Wheat Sourdough Loaf', quantity: 1, unit_price: 4.2, price: 4.2 },
    ],
  },
  confidences: [
    { field: 'vendor_name', label: 'Vendor Name', value: 'Metro Quick Supermarket', confidence: 97.5, status: 'valid', bbox: [5, 6, 11, 45] },
    { field: 'invoice_number', label: 'Receipt Number', value: 'CSH-9921', confidence: 96.0, status: 'valid', bbox: [5, 68, 9, 88] },
    { field: 'invoice_date', label: 'Receipt Date', value: '04/09/2026', confidence: 95.2, status: 'valid', bbox: [9, 68, 12, 88] },
    { field: 'gstin', label: 'GSTIN / Tax ID', value: null, confidence: 24.0, status: 'not_detected' },
    { field: 'state', label: 'State', value: null, confidence: 18.0, status: 'not_detected' },
    { field: 'state_code', label: 'State Code', value: null, confidence: 15.0, status: 'not_detected' },
    { field: 'total_tax', label: 'Sales Tax', value: '$ 0.56', confidence: 91.2, status: 'valid', bbox: [62, 55, 67, 88] },
    { field: 'total_amount', label: 'Total Paid', value: '$ 11.76', confidence: 98.4, status: 'valid', bbox: [68, 55, 74, 88] },
  ],
  qualityScore: 78.5,
  extractedFieldsCount: 5,
  totalFieldsCount: 8,
};

export const INITIAL_HISTORY: ExtractionResult[] = [
  MOCK_INVOICE_TOPSTAR,
  MOCK_INVOICE_VERTEX,
  MOCK_INVOICE_CORD,
  {
    id: 'doc-inv-001',
    fileName: 'Invoice_001.pdf',
    fileSize: '1.4 MB',
    fileType: 'application/pdf',
    fileUrl: MOCK_INVOICE_VERTEX.fileUrl,
    uploadedAt: '09/09/2026, 16:30',
    status: 'Completed',
    processingTimeSec: 2.1,
    modelName: 'Qwen2-VL-2B (QLoRA)',
    data: {
      vendor_name: 'Acme Hardware Supplies',
      invoice_number: 'ACM-771',
      invoice_date: '09/09/2026',
      gstin: '27AABCA1234D1Z2',
      state: 'Maharashtra',
      state_code: '27',
      total_tax: 2450,
      total_amount: 26950,
      line_items: [
        { id: '1', item: 'Heavy Duty Structural Fasteners', quantity: 200, unit_price: 100, price: 20000 },
        { id: '2', item: 'Industrial Sealant 500ml', quantity: 9, unit_price: 500, price: 4500 },
      ],
    },
    confidences: MOCK_INVOICE_VERTEX.confidences,
    qualityScore: 98.2,
    extractedFieldsCount: 8,
    totalFieldsCount: 8,
  },
  {
    id: 'doc-rec-002',
    fileName: 'Receipt_002.jpg',
    fileSize: '950 KB',
    fileType: 'image/jpeg',
    fileUrl: MOCK_INVOICE_CORD.fileUrl,
    uploadedAt: '08/09/2026, 14:15',
    status: 'Completed',
    processingTimeSec: 1.9,
    modelName: 'Qwen2-VL-2B (QLoRA)',
    data: MOCK_INVOICE_CORD.data,
    confidences: MOCK_INVOICE_CORD.confidences,
    qualityScore: 96.8,
    extractedFieldsCount: 8,
    totalFieldsCount: 8,
  },
  {
    id: 'doc-inv-003',
    fileName: 'Invoice_003_blurred.pdf',
    fileSize: '3.8 MB',
    fileType: 'application/pdf',
    fileUrl: MOCK_INVOICE_TOPSTAR.fileUrl,
    uploadedAt: '07/09/2026, 09:20',
    status: 'Failed',
    processingTimeSec: 0.8,
    modelName: 'Qwen2-VL-2B (QLoRA)',
    data: {
      vendor_name: null,
      invoice_number: null,
      invoice_date: null,
      gstin: null,
      state: null,
      state_code: null,
      total_tax: null,
      total_amount: null,
      line_items: [],
    },
    confidences: [],
    qualityScore: 0,
    extractedFieldsCount: 0,
    totalFieldsCount: 8,
  },
  MOCK_INVOICE_WARNING,
];

export const PIPELINE_STEPS = [
  { id: 'upload', name: 'Uploading Document', description: 'Transferring image payload to edge device runtime memory' },
  { id: 'ocr_preprocess', name: 'Reading & Vision Preprocessing', description: 'Normalizing DPI, dynamic resolution patch slicing & tokenization' },
  { id: 'vision_backbone', name: 'Running Multi-Modal OCR', description: 'High-density visual positional embeddings generated' },
  { id: 'qwen_qlora', name: 'Qwen2-VL + QLoRA Inference', description: 'Vision-Language cross attention over document key tokens' },
  { id: 'json_struct', name: 'Generating Structured JSON', description: 'Schema validation against CORD/Invoice extraction contract' },
];
