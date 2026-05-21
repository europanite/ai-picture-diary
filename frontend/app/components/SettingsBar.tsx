import React from "react";
import { View, Text, TouchableOpacity, useWindowDimensions, Linking, Alert, Platform } from "react-native";

const REPO_URL = "https://github.com/europanite/ai-picture-diary";

function Btn({ title, onPress }: { title: string; onPress: () => void }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={{ paddingVertical: 6, paddingHorizontal: 10, borderWidth: 1, borderRadius: 8, backgroundColor: "#fff" }}
      accessibilityRole="button"
      accessibilityLabel={title}
    >
      <Text style={{ fontWeight: "600" }}>{title}</Text>
    </TouchableOpacity>
  );
}

function resolveUrl(url: string): string {
  const s = String(url ?? "").trim();
  if (!s) return "";
  if (/^[a-z][a-z0-9+.-]*:/i.test(s) || s.startsWith("//")) return s;
  if (Platform.OS === "web" && typeof window !== "undefined") {
    return new URL(s, window.location.href).toString();
  }
  return s;
}

async function openUrl(url: string) {
  const target = resolveUrl(url);
  if (!target) return;
  try {
    if (Platform.OS === "web" && typeof window !== "undefined") {
      window.location.assign(target);
      return;
    }
    const ok = await Linking.canOpenURL(target);
    if (!ok) throw new Error("canOpenURL returned false");
    await Linking.openURL(target);
  } catch {
    Alert.alert("Open link failed", `Could not open:\n${target}`);
  }
}

type Props = { title?: string };

export default function SettingsBar({ title = "ai-picture-diary" }: Props) {
  const { width } = useWindowDimensions();
  const isNarrow = width < 520;

  return (
    <View
      style={{
        padding: 12,
        borderBottomWidth: 1,
        backgroundColor: "#333366",
      }}
    >
      {isNarrow ? (
        // NARROW: stack (avoid overlap)
        <View style={{ 
          gap: 8, 
          padding: 6,
          alignItems: "center"
        }}>
          <Text style={{ 
            fontSize: 32, 
            fontWeight: "512", 
            color: "#fff"
            }}>
              {title}
          </Text>
        </View>
      ) : (
        // WIDE: title pinned to center
        <View style={{ 
          position: "relative", 
          justifyContent: "center", 
          padding: 6,
          minHeight: 28 }}>

          {/* Center title (overlay) */}
          <View
            pointerEvents="none"
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: 0,
              bottom: 0,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Text style={{ 
              fontSize: 32, 
              fontWeight: "512", 
              color: 
              "#fff" 
            }}>{title}</Text>
          </View>
        </View>
      )}
    </View>
  );
}
