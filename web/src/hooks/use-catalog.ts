"use client";

import { useEffect, useState } from "react";
import type { Catalog } from "@/lib/catalog";
import { loadCatalog } from "@/lib/catalog";

export function useCatalog() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCatalog()
      .then(setCatalog)
      .catch((err: Error) => setError(err.message));
  }, []);

  return { catalog, error, loading: !catalog && !error };
}
