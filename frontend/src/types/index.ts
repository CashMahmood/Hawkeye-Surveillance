export interface Box {
    cls: number;
    label: string;
    conf: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}

export interface DetectionMessage {
    timestamp: string;
    fps: number;
    counts: {
        persons: number;
        weapons: number;
    };
    threats: string[];
    boxes: Box[];
    status: string;
    debug: {
        model_used: string;
    };
}

export interface SurveillanceEvent {
    id: number;
    timestamp: string;
    type: string;
    labels: string; // JSON string
    confidence: number;
    image_path: string;
    bboxes: string; // JSON string
}
