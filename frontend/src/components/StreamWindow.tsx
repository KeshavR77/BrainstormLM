import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, Terminal } from 'lucide-react';

interface Props {
  text: string;
  isStreaming: boolean;
  label?: string;
}

export function StreamWindow({ text, isStreaming, label }: Props) {
  const [displayText, setDisplayText] = useState('');
  const [copied, setCopied] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const charIdx = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setDisplayText(text);
      charIdx.current = text.length;
      return;
    }

    charIdx.current = 0;
    setDisplayText('');

    const interval = setInterval(() => {
      charIdx.current += 2;
      if (charIdx.current >= text.length) {
        setDisplayText(text);
        clearInterval(interval);
      } else {
        setDisplayText(text.slice(0, charIdx.current));
      }
    }, 18);

    return () => clearInterval(interval);
  }, [text, isStreaming]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [displayText]);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="stream-window">
      <div className="stream-window__header">
        <div className="stream-window__header-left">
          <Terminal size={13} />
          <span>{label ?? 'Steered Generation'}</span>
        </div>
        <div className="stream-window__header-right">
          <AnimatePresence mode="wait">
            {isStreaming && charIdx.current < text.length && (
              <motion.span
                className="stream-window__live"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span className="stream-window__live-dot" />
                LIVE
              </motion.span>
            )}
          </AnimatePresence>
          <button className="stream-window__copy" onClick={handleCopy} aria-label="Copy">
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        </div>
      </div>
      <div className="stream-window__body" ref={scrollRef}>
        <p className="stream-window__text">
          {displayText}
          {isStreaming && charIdx.current < text.length && (
            <span className="stream-window__cursor" />
          )}
        </p>
      </div>
    </div>
  );
}
