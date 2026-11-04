"use client";

import { buildingsToGeoJson, eventOrigin, type DamageFeatureCollection } from "@/lib/geo";
import type { DamageClass, DisasterEvent } from "@/lib/catalog";
import { useEffect } from "react";
import { CircleMarker, GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";
import type { Layer, PathOptions } from "leaflet";
import "leaflet/dist/leaflet.css";

const COLORS: Record<DamageClass, string> = {
  "no-damage": "#22c55e",
  "minor-damage": "#eab308",
  "major-damage": "#f97316",
  destroyed: "#ef4444",
};

function FitAll({ events }: { events: DisasterEvent[] }) {
  const map = useMap();
  useEffect(() => {
    const points = events
      .map((item) => eventOrigin(item))
      .filter((point): point is { latitude: number; longitude: number } => point !== null)
      .map((point) => [point.latitude, point.longitude] as [number, number]);
    if (points.length > 0) map.fitBounds(points, { padding: [36, 36], maxZoom: 3 });
  }, [map, events]);
  return null;
}

function ResizeMap() {
  const map = useMap();
  useEffect(() => {
    const id = window.setTimeout(() => map.invalidateSize(), 50);
    return () => window.clearTimeout(id);
  }, [map]);
  return null;
}

function FlyTo({ latitude, longitude }: { latitude: number; longitude: number }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo([latitude, longitude], 16, { duration: 0.6 });
  }, [map, latitude, longitude]);
  return null;
}

export function GisMap({
  events,
  event,
  showPred,
  selectedId,
  showAll,
  onSelectBuilding,
  onSelectEvent,
}: {
  events: DisasterEvent[];
  event: DisasterEvent;
  showPred: boolean;
  selectedId: string | null;
  showAll: boolean;
  onSelectBuilding: (uid: string) => void;
  onSelectEvent: (id: string) => void;
}) {
  const origin = eventOrigin(event);
  const data = buildingsToGeoJson(event, showPred);
  if (!origin) {
    return <p className="p-4 text-sm text-red-300">This event has no map location.</p>;
  }

  return (
    <MapContainer
      center={[origin.latitude, origin.longitude]}
      zoom={16}
      style={{ height: 560, width: "100%" }}
      scrollWheelZoom
    >
      <ResizeMap />
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {showAll ? (
        <FitAll events={events} />
      ) : (
        <FlyTo latitude={origin.latitude} longitude={origin.longitude} />
      )}
      {events.map((item) => {
        const point = eventOrigin(item);
        if (!point || (!showAll && item.id === event.id)) return null;
        return (
          <CircleMarker
            key={item.id}
            center={[point.latitude, point.longitude]}
            radius={8}
            pathOptions={{ color: "#fb923c", fillColor: "#fb923c", fillOpacity: 0.85, weight: 1 }}
            eventHandlers={{ click: () => onSelectEvent(item.id) }}
          />
        );
      })}
      {!showAll && (
      <GeoJSON
        key={`${event.id}-${showPred}-${selectedId}`}
        data={data as DamageFeatureCollection}
        style={(feature): PathOptions => {
          const damage = (feature?.properties?.damage ?? "no-damage") as DamageClass;
          const active = feature?.properties?.uid === selectedId;
          return {
            color: COLORS[damage],
            weight: active ? 3 : 1.5,
            fillColor: COLORS[damage],
            fillOpacity: active ? 0.65 : 0.4,
          };
        }}
        onEachFeature={(feature, layer: Layer) => {
          const uid = feature.properties?.uid as string | undefined;
          if (!uid) return;
          layer.on({ click: () => onSelectBuilding(uid) });
        }}
      />
      )}
    </MapContainer>
  );
}
