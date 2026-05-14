/**
 * ShortcutsSheet — modal cheatsheet, opens on `?` key.
 */
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Group { title: string; items: { keys: string[]; label: string }[] }

const GROUPS: Group[] = [
  {
    title: 'Workspace',
    items: [
      { keys: ['⌘', 'K'], label: 'Command palette' },
      { keys: ['⌘', '/'], label: 'Toggle layers panel' },
      { keys: ['⌘', '.'], label: 'Toggle inspector' },
      { keys: ['?'],      label: 'Show this sheet' },
    ],
  },
  {
    title: 'Tools',
    items: [
      { keys: ['V'], label: 'Select' },
      { keys: ['E'], label: 'Expand' },
      { keys: ['B'], label: 'Branch (manual child)' },
      { keys: ['F'], label: 'Frame / group' },
      { keys: ['C'], label: 'Comment' },
      { keys: ['T'], label: 'Text annotation' },
      { keys: ['H'], label: 'Hand / pan' },
      { keys: ['Z'], label: 'Zoom' },
    ],
  },
  {
    title: 'Canvas',
    items: [
      { keys: ['⌘', '0'],         label: 'Fit view' },
      { keys: ['⌘', '1'],         label: 'Zoom 100%' },
      { keys: ['⌘', 'Enter'],     label: 'Expand selected node' },
      { keys: ['⌘', 'D'],         label: 'Duplicate selected' },
      { keys: ['⌘', 'G'],         label: 'Group into frame' },
      { keys: ['Space', '+drag'], label: 'Pan canvas' },
      { keys: ['['],              label: 'Cycle to previous sibling' },
      { keys: [']'],              label: 'Cycle to next sibling' },
    ],
  },
  {
    title: 'History',
    items: [
      { keys: ['⌘', 'Z'],         label: 'Undo' },
      { keys: ['⌘', '⇧', 'Z'],   label: 'Redo' },
      { keys: ['⌘', 'S'],         label: 'Save snapshot' },
    ],
  },
];

export function ShortcutsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={onClose}
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink-0/30 backdrop-blur-sm p-6"
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl rounded-2xl bg-paper border border-[var(--ink-line-bold)] shadow-lift overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--ink-line)]">
              <h2
                className="font-display text-ink-0 text-[22px] tracking-tight"
                style={{ fontVariationSettings: '"wght" 520, "opsz" 48' }}
              >
                Keyboard shortcuts
              </h2>
              <button
                onClick={onClose}
                className="text-ink-2 hover:text-ink-0 transition-colors p-1 rounded-md hover:bg-paper-2"
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-6 grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6 max-h-[70vh] overflow-y-auto">
              {GROUPS.map((g) => (
                <section key={g.title}>
                  <div className="text-[10px] uppercase tracking-[0.18em] text-ink-3 mb-3">{g.title}</div>
                  <ul className="space-y-1.5">
                    {g.items.map((it) => (
                      <li key={it.label} className="flex items-center justify-between gap-3 py-1">
                        <span className="text-[12.5px] text-ink-1">{it.label}</span>
                        <div className="flex items-center gap-1 shrink-0">
                          {it.keys.map((k, i) => (
                            <kbd
                              key={i}
                              className="px-1.5 py-0.5 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[10.5px] font-mono text-ink-0 tabular-nums"
                            >
                              {k}
                            </kbd>
                          ))}
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
