import React, { useEffect, useState, useMemo } from 'react';
import { Calendar, RefreshCw, X, Filter, Clock, Fingerprint } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import type { SurveillanceEvent } from '../types';

const Events: React.FC = () => {
    const [events, setEvents] = useState<SurveillanceEvent[]>([]);
    const [selectedEvent, setSelectedEvent] = useState<SurveillanceEvent | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [filterType, setFilterType] = useState<string>('ALL');

    const fetchEvents = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get('http://localhost:8000/events');
            setEvents(response.data);
        } catch (error) {
            console.error("Failed to fetch events:", error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchEvents();
        const interval = setInterval(fetchEvents, 10000); // Auto-refresh every 10s
        return () => clearInterval(interval);
    }, []);

    // Tactical log displays all recorded priority incidents

    const formatDate = (dateStr: string) => {
        return new Date(dateStr).toLocaleString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    };

    return (
        <div className="flex flex-col h-screen p-4 bg-background text-tactical-green font-mono overflow-hidden">
            {/* Header Bar */}
            <header className="flex justify-between items-center border-b border-tactical-green/30 pb-3 mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 border border-tactical-green/30 rounded-sm">
                        <Clock className="w-8 h-8 neon-text-green" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-black tracking-tighter uppercase italic leading-none">
                            INCIDENT <span className="text-tactical-green/40">Archive</span>
                        </h1>
                        <p className="text-[10px] font-bold tracking-[0.2em] mt-1 opacity-60 uppercase">Operational History Log</p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={fetchEvents}
                        className="btn-tactical flex items-center gap-2 h-10 px-4"
                    >
                        <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        RE-INDEX DATABASE
                    </button>
                </div>
            </header>

            {/* Grid View */}
            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {isLoading && events.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full opacity-20 animate-pulse">
                        <RefreshCw className="w-16 h-16 mb-4 animate-spin" />
                        <p className="uppercase tracking-[0.5em] font-black">Scanning Storage Media...</p>
                    </div>
                ) : events.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full opacity-30">
                        <Calendar className="w-16 h-16 mb-4" />
                        <p className="uppercase tracking-[0.2em] font-black">Zero Priority Incidents Found</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4 pb-8">
                        <AnimatePresence>
                            {events.map(event => (
                                <motion.div
                                    key={event.id}
                                    layout
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    whileHover={{ y: -4 }}
                                    className="panel group cursor-pointer hover:border-tactical-green/60 transition-all overflow-hidden border-tactical-green/20"
                                    onClick={() => setSelectedEvent(event)}
                                >
                                    <div className="relative aspect-video bg-black overflow-hidden border-b border-tactical-green/10">
                                        <img
                                            src={`http://localhost:8000/images/${event.image_path}?t=${new Date(event.timestamp).getTime()}`}
                                            alt={event.type}
                                            className="w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-60" />
                                        <div className={`absolute top-2 right-2 px-2 py-0.5 text-[9px] font-black tracking-widest border ${event.type === 'WEAPON_DETECTED' || event.type === 'PERSON_WITH_WEAPON'
                                            ? 'border-tactical-red bg-tactical-red/20 text-tactical-red'
                                            : 'border-tactical-amber bg-tactical-amber/20 text-tactical-amber'
                                            }`}>
                                            {event.type.replace(/_/g, ' ')}
                                        </div>
                                    </div>
                                    <div className="p-3 space-y-2 bg-panel-bg/30">
                                        <div className="flex justify-between items-center text-[9px] font-black uppercase tracking-tighter text-tactical-green/40">
                                            <span>REF: #{event.id.toString().padStart(4, '0')}</span>
                                            <div className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                <span>{formatDate(event.timestamp)}</span>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-[10px] font-bold">
                                            <Fingerprint className="w-3 h-3 opacity-50" />
                                            <span className="truncate opacity-80 italic">SIGNATURES: {JSON.parse(event.labels).join(', ')}</span>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                )}
            </div>

            {/* Detail Modal */}
            <AnimatePresence>
                {selectedEvent && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-50 flex items-center justify-center p-4 lg:p-12 bg-black/95 backdrop-blur-md"
                    >
                        <motion.div
                            initial={{ scale: 0.95, y: 20 }}
                            animate={{ scale: 1, y: 0 }}
                            className="panel w-full max-w-7xl max-h-full flex flex-col overflow-hidden border-tactical-green/30 shadow-[0_0_50px_rgba(0,255,65,0.1)]"
                        >
                            <div className="panel-header flex justify-between items-center p-4 bg-panel-border/30">
                                <div className="flex items-center gap-4">
                                    <span className="bg-tactical-red/20 border border-tactical-red/50 px-3 py-1 text-tactical-red font-black text-xs">CRITICAL DATA PACKET</span>
                                    <span className="text-sm font-black tracking-[0.2em] opacity-80">INCIDENT ID: {selectedEvent.id.toString().padStart(6, '0')}</span>
                                </div>
                                <button
                                    onClick={() => setSelectedEvent(null)}
                                    className="p-1 hover:text-tactical-red transition-colors active:scale-90"
                                >
                                    <X className="w-8 h-8" />
                                </button>
                            </div>

                            <div className="flex flex-col lg:flex-row flex-1 overflow-hidden">
                                <div className="lg:w-3/4 bg-black flex items-center justify-center p-4 border-r border-tactical-green/10 relative overflow-hidden">
                                    {/* Background Grid Pattern */}
                                    <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle, #00ff41 1px, transparent 1px)', backgroundSize: '20px 20px' }} />

                                    <img
                                        src={`http://localhost:8000/images/${selectedEvent.image_path}?t=${new Date(selectedEvent.timestamp).getTime()}`}
                                        alt="Full Evidence"
                                        className="max-h-full max-w-full object-contain shadow-[0_0_30px_rgba(0,0,0,0.5)] border border-white/5 z-10"
                                    />

                                    {/* Tactical Decorations */}
                                    <div className="absolute top-8 left-8 w-16 h-16 border-t-2 border-l-2 border-tactical-red z-20 opacity-60" />
                                    <div className="absolute bottom-8 right-8 w-16 h-16 border-b-2 border-r-2 border-tactical-red z-20 opacity-60" />
                                    <div className="absolute top-1/2 left-4 w-6 h-[1px] bg-tactical-red/40 z-20" />
                                    <div className="absolute top-1/2 right-4 w-6 h-[1px] bg-tactical-red/40 z-20" />
                                </div>

                                <div className="lg:w-1/4 p-6 space-y-8 overflow-y-auto bg-panel-bg/80 backdrop-blur-xl">
                                    <div className="space-y-6">
                                        <div>
                                            <h3 className="text-[10px] uppercase tracking-[0.2em] text-tactical-green/40 font-black border-b border-tactical-green/10 pb-1 mb-2">Capture Marker</h3>
                                            <p className="text-2xl font-black tracking-tighter text-white">{formatDate(selectedEvent.timestamp)}</p>
                                        </div>

                                        <div>
                                            <h3 className="text-[10px] uppercase tracking-[0.2em] text-tactical-green/40 font-black border-b border-tactical-green/10 pb-1 mb-2">Classification</h3>
                                            <p className="text-2xl font-black text-tactical-red shadow-glow-red italic tracking-tight">{selectedEvent.type.replace(/_/g, ' ')}</p>
                                        </div>

                                        <div>
                                            <h3 className="text-[10px] uppercase tracking-[0.2em] text-tactical-green/40 font-black border-b border-tactical-green/10 pb-1 mb-2">Confidence Level</h3>
                                            <div className="mt-3">
                                                <div className="flex justify-between mb-1 text-[10px] font-black">
                                                    <span className="opacity-60 uppercase">Reliability Index</span>
                                                    <span className="text-white">{(selectedEvent.confidence * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-white/5 h-3 rounded-sm p-0.5 border border-white/10">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${selectedEvent.confidence * 100}%` }}
                                                        className={`h-full ${selectedEvent.confidence > 0.7 ? 'bg-tactical-green' : 'bg-tactical-amber'} shadow-glow`}
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        <div>
                                            <h3 className="text-[10px] uppercase tracking-[0.2em] text-tactical-green/40 font-black border-b border-tactical-green/10 pb-1 mb-2">AI Segment Results</h3>
                                            <div className="flex flex-wrap gap-2 pt-2">
                                                {JSON.parse(selectedEvent.labels).map((label: string, i: number) => (
                                                    <span key={i} className={`px-2 py-1 border text-[10px] font-black uppercase rounded-sm ${['gun', 'knife', 'pistol', 'rifle'].some(w => label.toLowerCase().includes(w))
                                                        ? 'bg-tactical-red/10 border-tactical-red text-tactical-red'
                                                        : 'bg-tactical-green/10 border-tactical-green text-tactical-green'
                                                        }`}>
                                                        {label}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="pt-8 border-t border-tactical-green/10">
                                        <button
                                            onClick={() => setSelectedEvent(null)}
                                            className="w-full h-12 border-2 border-tactical-green bg-tactical-green/5 hover:bg-tactical-green hover:text-black transition-all font-black uppercase tracking-[0.3em] active:scale-95 shadow-[0_0_15px_rgba(0,255,65,0.1)]"
                                        >
                                            Acknowledge Log
                                        </button>
                                        <p className="text-[9px] text-center mt-3 opacity-40 uppercase tracking-widest">Operator signature required</p>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Events;
