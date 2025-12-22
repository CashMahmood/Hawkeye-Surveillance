import React, { useEffect, useState, useRef } from 'react';
import { Shield, AlertTriangle, Users, Target, Activity, Radio } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DetectionMessage } from '../types';

const Dashboard: React.FC = () => {
    const [detection, setDetection] = useState<DetectionMessage | null>(null);
    const [logs, setLogs] = useState<{ id: number; msg: string; type: 'info' | 'warn' | 'error' }[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const videoRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        let ws: WebSocket;
        const connect = () => {
            ws = new WebSocket('ws://localhost:8000/ws/detections');

            ws.onopen = () => {
                setIsConnected(true);
                addLog("Neural Uplink Synchronized", 'info');
            };

            ws.onclose = () => {
                setIsConnected(false);
                addLog("Neurolink Offline - Attempting Reconnection", 'error');
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const data: DetectionMessage = JSON.parse(event.data);
                setDetection(data);

                if (data.threats.length > 0) {
                    data.threats.forEach(threat => {
                        // Check if threat was already in recent logs to avoid spam
                        const lastLog = logs[0];
                        if (!lastLog || !lastLog.msg.includes(threat)) {
                            addLog(`CRITICAL: ${threat.replace(/_/g, ' ')}`, 'error');
                        }
                    });
                }
            };
        };

        connect();
        return () => ws.close();
    }, []);

    const addLog = (msg: string, type: 'info' | 'warn' | 'error') => {
        setLogs(prev => [{ id: Date.now(), msg, type }, ...prev].slice(0, 20));
    };

    useEffect(() => {
        if (detection && canvasRef.current && videoRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const img = videoRef.current;
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;

            const scaleX = canvas.width / 640;
            const scaleY = canvas.height / 480;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            detection.boxes.forEach(box => {
                const isWeapon = box.label.toLowerCase().includes('person') ? false : true;
                const color = isWeapon ? '#ff0000' : '#00ff41';

                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                ctx.strokeRect(
                    box.x1 * scaleX,
                    box.y1 * scaleY,
                    (box.x2 - box.x1) * scaleX,
                    (box.y2 - box.y1) * scaleY
                );

                // Label bg
                ctx.fillStyle = `${color}cc`;
                const labelText = `${box.label.toUpperCase()} [${Math.round(box.conf * 100)}%]`;
                const textWidth = ctx.measureText(labelText).width;
                ctx.fillRect(box.x1 * scaleX, box.y1 * scaleY - 15, textWidth + 10, 15);

                ctx.fillStyle = '#000000';
                ctx.font = 'bold 10px JetBrains Mono, monospace';
                ctx.fillText(labelText, box.x1 * scaleX + 5, box.y1 * scaleY - 4);
            });
        }
    }, [detection]);

    return (
        <div className="flex flex-col h-screen p-4 bg-background text-tactical-green font-mono overflow-hidden">
            {/* Header Bar */}
            <header className="flex justify-between items-center border-b border-tactical-green/30 pb-3 mb-4">
                <div className="flex items-center gap-3">
                    <div className={`p-1 border ${isConnected ? 'border-tactical-green' : 'border-tactical-red'} rounded-sm`}>
                        <Shield className={`w-8 h-8 ${isConnected ? 'neon-text-green' : 'neon-text-red'}`} />
                    </div>
                    <div>
                        <h1 className="text-2xl font-black tracking-tighter uppercase italic leading-none">
                            HAWKEYE <span className="text-tactical-green/40">Surveillance</span>
                        </h1>
                        <div className="flex items-center gap-2 mt-1">
                            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-tactical-green animate-pulse' : 'bg-tactical-red'}`} />
                            <span className="text-[10px] font-bold tracking-[0.2em]">{isConnected ? "ONLINE // LINKED" : "OFFLINE // SEEKING"}</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-8 text-[11px] font-bold tracking-widest">
                    <div className="flex flex-col items-end">
                        <span className="text-tactical-green/40 uppercase text-[9px]">Ingest Rate</span>
                        <div className="flex items-center gap-1">
                            <Activity className="w-3 h-3" />
                            <span>{detection?.fps || 0} FPS</span>
                        </div>
                    </div>
                    <div className="flex flex-col items-end">
                        <span className="text-tactical-green/40 uppercase text-[9px]">System Time</span>
                        <div className="bg-tactical-green/10 px-3 py-1 border border-tactical-green/20 rounded-sm">
                            {new Date().toLocaleTimeString([], { hour12: false })}
                        </div>
                    </div>
                </div>
            </header>

            {/* Main HUD */}
            <div className="flex flex-1 gap-4 overflow-hidden mb-2">

                {/* Left Column: Comms & Intel */}
                <div className="w-80 flex flex-col gap-4">
                    <div className="panel flex-1 flex flex-col border-tactical-green/20">
                        <div className="panel-header flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Radio className="w-4 h-4" />
                                <span>COMMS_LOG</span>
                            </div>
                            <div className="text-[9px] opacity-40">CH_01</div>
                        </div>
                        <div className="flex-1 p-3 overflow-y-auto font-mono text-[10px] space-y-2 bg-black/40">
                            <AnimatePresence initial={false}>
                                {logs.map(log => (
                                    <motion.div
                                        key={log.id}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        className={`flex gap-2 ${log.type === 'error' ? 'text-tactical-red' :
                                            log.type === 'warn' ? 'text-tactical-amber' : 'text-tactical-green'
                                            }`}
                                    >
                                        <span className="opacity-40 whitespace-nowrap">[{new Date(log.id).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}]</span>
                                        <span className="font-bold">{log.msg}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>

                {/* Center: Tactical Optical Feed */}
                <div className="flex-1 flex flex-col gap-4">
                    <div className="panel flex-1 relative bg-black/80 overflow-hidden border-tactical-green/30 group">
                        {/* Feed Metadata Overlay */}
                        <div className="absolute top-4 left-4 z-10 flex gap-2">
                            <div className="bg-black/90 px-3 py-1 text-[10px] border border-tactical-green/30 font-bold backdrop-blur-md">
                                FEED_TYPE: PRIMARY_OPTICAL
                            </div>
                            <div className="bg-black/90 px-3 py-1 text-[10px] border border-tactical-amber/30 text-tactical-amber font-bold backdrop-blur-md italic">
                                WEIGHTS: {detection?.debug.model_used || "WAITING..."}
                            </div>
                        </div>

                        {/* Scale Indicator */}
                        <div className="absolute bottom-4 right-4 z-10 bg-black/60 px-2 py-1 border border-tactical-green/20 text-[9px]">
                            RESOLUTION: 640x480
                        </div>

                        {/* Legend */}
                        <div className="absolute bottom-4 left-4 z-10 flex gap-4 bg-black/60 p-2 border border-tactical-green/20">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-tactical-green" />
                                <span className="text-[9px] uppercase font-bold">Person</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-tactical-red" />
                                <span className="text-[9px] uppercase font-bold">Threat Object</span>
                            </div>
                        </div>

                        {/* The actual video feed */}
                        <div className="w-full h-full flex items-center justify-center bg-black">
                            {isConnected ? (
                                <>
                                    <img
                                        ref={videoRef}
                                        src="http://localhost:8000/video"
                                        className="max-h-full max-w-full object-contain"
                                        alt="Live Feed"
                                    />
                                    <canvas
                                        ref={canvasRef}
                                        className="absolute pointer-events-none"
                                        style={{
                                            width: videoRef.current?.clientWidth,
                                            height: videoRef.current?.clientHeight,
                                        }}
                                    />
                                </>
                            ) : (
                                <div className="flex flex-col items-center gap-4 opacity-20 animate-pulse">
                                    <Target className="w-24 h-24" />
                                    <p className="text-xl font-black uppercase tracking-[0.3em]">No Visual Link</p>
                                </div>
                            )}
                        </div>

                        {/* Tactical Corners */}
                        <div className="absolute top-0 left-0 w-12 h-12 border-t-2 border-l-2 border-tactical-green opacity-40" />
                        <div className="absolute top-0 right-0 w-12 h-12 border-t-2 border-r-2 border-tactical-green opacity-40" />
                        <div className="absolute bottom-0 left-0 w-12 h-12 border-b-2 border-l-2 border-tactical-green opacity-40" />
                        <div className="absolute bottom-0 right-0 w-12 h-12 border-b-2 border-r-2 border-tactical-green opacity-40" />

                        {/* Scanning Line Effect */}
                        <div className="absolute top-0 left-0 w-full h-[1px] bg-tactical-green/20 animate-scan z-0 pointer-events-none" />
                    </div>

                    {/* Bottom Alert Strip */}
                    <div className="h-20 panel flex items-center justify-between px-6 bg-tactical-green/5 overflow-hidden">
                        <AnimatePresence mode="wait">
                            {detection?.threats.length ? (
                                <motion.div
                                    key="alert"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    className="flex items-center gap-6 w-full"
                                >
                                    <div className="bg-tactical-red p-3 animate-pulse shadow-glow-red rounded-sm">
                                        <AlertTriangle className="w-8 h-8 text-black" />
                                    </div>
                                    <div className="flex-1">
                                        <h2 className="text-tactical-red font-black text-2xl tracking-tighter uppercase leading-none">
                                            CRITICAL_THREAT_DETECTED
                                        </h2>
                                        <p className="text-tactical-red/60 text-sm font-bold mt-1">
                                            TYPE: {detection.threats.join(' + ').replace(/_/g, ' ')}
                                        </p>
                                    </div>
                                    <div className="text-tactical-red font-black text-4xl animate-pulse italic">
                                        ! ! !
                                    </div>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="idle"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex items-center gap-4 opacity-40"
                                >
                                    <Shield className="w-10 h-10" />
                                    <div>
                                        <h2 className="text-xl font-black uppercase tracking-widest leading-none">Sector_Clear</h2>
                                        <p className="text-[10px] mt-1 uppercase font-bold tracking-widest leading-none">Scanning Peripheral Environment</p>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Right Column: Status Matrix */}
                <div className="w-72 flex flex-col gap-4">
                    <div className="panel p-5 space-y-6 flex-1">
                        <div className="panel-header -mx-5 -mt-5 mb-5 flex items-center gap-2">
                            <Activity className="w-4 h-4" />
                            <span>STATUS_MATRIX</span>
                        </div>

                        <TargetCard label="INDIVIDUALS" count={detection?.counts.persons || 0} icon={Users} color="green" />
                        <TargetCard label="THREAT_SIGS" count={detection?.counts.weapons || 0} icon={Target} color="red" />

                        <div className="pt-4 space-y-3 border-t border-tactical-green/20">
                            <h3 className="text-[10px] font-black uppercase tracking-widest text-tactical-green/40">Subsystem Protocol</h3>
                            <ProtocolRow label="GRP_ANALYTICS" status="ACTIVE" />
                            <ProtocolRow label="PROXIMITY_MAPPING" status="ACTIVE" />
                            <ProtocolRow label="SNAPSHOT_AUTO" status="STANDBY" />
                        </div>

                        <div className="mt-8">
                            <div className="text-[10px] uppercase font-bold text-tactical-green/40 mb-2">Threat Legend</div>
                            <div className="space-y-1">
                                <LegendItem color="bg-tactical-green" label="Neutral Contact" />
                                <LegendItem color="bg-tactical-red" label="Weapon Identified" />
                                <LegendItem color="bg-tactical-amber" label="Suspicious Group" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const TargetCard = ({ label, count, icon: Icon, color }: any) => (
    <div className={`p-4 border ${color === 'red' ? 'border-tactical-red/30 bg-tactical-red/5' : 'border-tactical-green/20 bg-tactical-green/5'} rounded-sm`}>
        <div className="flex items-center justify-between mb-2">
            <Icon className={`w-4 h-4 ${color === 'red' ? 'text-tactical-red shadow-glow-red' : 'text-tactical-green'}`} />
            <span className={`text-[10px] font-bold tracking-widest uppercase ${color === 'red' ? 'text-tactical-red/60' : 'text-tactical-green/60'}`}>{label}</span>
        </div>
        <div className={`text-5xl font-black tracking-tighter ${color === 'red' ? 'neon-text-red' : 'neon-text-green'}`}>{count}</div>
    </div>
);

const ProtocolRow = ({ label, status }: any) => (
    <div className="flex justify-between items-center text-[10px] font-bold">
        <span className="opacity-60">{label}</span>
        <span className="text-tactical-green tracking-tighter shadow-glow px-2 py-0.5 border border-tactical-green/30 bg-tactical-green/5 font-black">{status}</span>
    </div>
);

const LegendItem = ({ color, label }: any) => (
    <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${color}`} />
        <span className="text-[9px] font-bold uppercase tracking-widest opacity-60">{label}</span>
    </div>
);

export default Dashboard;
