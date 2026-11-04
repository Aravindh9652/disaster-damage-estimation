import type { Building, DamageClass, DisasterEvent } from "@/lib/catalog";

export const METERS_PER_PIXEL = 0.8;

export const ANCHORS: Record<string, { latitude: number; longitude: number }> = {
  "hurricane-harvey-houston": { latitude: 29.7604, longitude: -95.3698 },
  "santa-rosa-wildfire": { latitude: 38.4404, longitude: -122.7141 },
  "palu-tsunami": { latitude: -0.8917, longitude: 119.8707 },
  "mexico-city-earthquake": { latitude: 19.4326, longitude: -99.1332 },
};

export function eventOrigin(event: DisasterEvent) {
  if (event.latitude != null && event.longitude != null) {
    return { latitude: event.latitude, longitude: event.longitude };
  }
  return ANCHORS[event.id] ?? null;
}

export function pixelToLonLat(x: number, y: number, event: DisasterEvent): [number, number] | null {
  const origin = eventOrigin(event);
  if (!origin) return null;
  const east = (x - event.width / 2) * METERS_PER_PIXEL;
  const north = (event.height / 2 - y) * METERS_PER_PIXEL;
  const latitude = origin.latitude + north / 111320;
  const longitude = origin.longitude + east / (111320 * Math.cos((origin.latitude * Math.PI) / 180));
  return [longitude, latitude];
}

export type DamageFeatureCollection = {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    geometry: { type: "Polygon"; coordinates: [number, number][][] };
    properties: { uid: string; damage: DamageClass };
  }[];
};

export function buildingsToGeoJson(event: DisasterEvent, showPred: boolean): DamageFeatureCollection {
  return {
    type: "FeatureCollection",
    features: event.buildings.flatMap((building) => {
      const ring = building.polygon
        .map((point) => pixelToLonLat(point.x, point.y, event))
        .filter((point): point is [number, number] => point !== null);
      if (ring.length < 3) return [];
      const closed = ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1]
        ? ring
        : [...ring, ring[0]];
      return [
        {
          type: "Feature" as const,
          geometry: { type: "Polygon" as const, coordinates: [closed] },
          properties: { uid: building.uid, damage: showPred ? building.pred : building.label },
        },
      ];
    }),
  };
}

export function buildingById(event: DisasterEvent, uid: string): Building | undefined {
  return event.buildings.find((building) => building.uid === uid);
}
