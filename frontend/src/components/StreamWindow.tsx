import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, Terminal } from 'lucide-react';

interface Props {
  text: string;
  isStreaming: boolean;
  label?: string;
}

export function StreamWindow({ text, isStreaming, label }: Props) {
  const [copied, setCopied] = useState(false);

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
            {isStreaming && (
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
      <div className="stream-window__body">
        <p className="stream-window__text">
          {text}
          {isStreaming && (
            <span className="stream-window__cursor" />
          )}
        </p>
      </div>
    </div>
  );
}
