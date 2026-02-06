import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Debounce a value - delays updating until after the specified delay
 * has passed since the last change.
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Returns a debounced callback that delays invoking the function
 * until after the specified delay has passed since the last call.
 */
export function useDebouncedCallback<T extends (...args: Parameters<T>) => void>(
  callback: T,
  delay: number
): T {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const callbackRef = useRef(callback);

  // Keep callback ref up to date
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const debouncedCallback = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        callbackRef.current(...args);
      }, delay);
    },
    [delay]
  ) as T;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback;
}

/**
 * A hook for controlled inputs that debounces the onChange callback
 * while keeping the input responsive. Returns [localValue, setLocalValue, debouncedValue].
 */
export function useDebouncedInput<T>(
  externalValue: T,
  onChange: (value: T) => void,
  delay: number = 300
): [T, (value: T) => void] {
  const [localValue, setLocalValue] = useState<T>(externalValue);
  const isInternalUpdate = useRef(false);

  // Sync external value changes (e.g., from presets)
  useEffect(() => {
    if (!isInternalUpdate.current) {
      setLocalValue(externalValue);
    }
    isInternalUpdate.current = false;
  }, [externalValue]);

  // Debounced callback to parent
  const debouncedOnChange = useDebouncedCallback(onChange, delay);

  const handleChange = useCallback(
    (value: T) => {
      isInternalUpdate.current = true;
      setLocalValue(value);
      debouncedOnChange(value);
    },
    [debouncedOnChange]
  );

  return [localValue, handleChange];
}
