import React from 'react';
import { NavigationContainer } from '@react-navigation/nnative';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Provider as PaperProvider } from 'react-native-paper';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Import screens
import LoginScreen from './src/screens/LoginScreen';
import CaptureScreen from './src/screens/CaptureScreen';
import ReviewScreen from './src/screens/ReviewScreen';
import DiaryScreen from './src/screens/DiaryScreen';
import AnalyticsScreen from './src/screens/AnalyticsScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Import components
import CustomDrawerContent from './src/components/CustomDrawerContent';
import TabBarIcon from './src/components/TabBarIcon';

// Import stores
import { useAuthStore } from './src/store/auth';

// Create navigators
const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();
const Drawer = createDrawerNavigator();

// Main tab navigator
function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => (
          <TabBarIcon route={route} focused={focused} color={color} size={size} />
        ),
        tabBarActiveTintColor: '#1f77b4',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}
    >
      <Tab.Screen 
        name="Capture" 
        component={CaptureScreen}
        options={{ title: '📸 Capture' }}
      />
      <Tab.Screen 
        name="Diary" 
        component={DiaryScreen}
        options={{ title: '📝 Diary' }}
      />
      <Tab.Screen 
        name="Analytics" 
        component={AnalyticsScreen}
        options={{ title: '📊 Analytics' }}
      />
    </Tab.Navigator>
  );
}

// Main drawer navigator
function MainDrawerNavigator() {
  return (
    <Drawer.Navigator
      drawerContent={(props) => <CustomDrawerContent {...props} />}
      screenOptions={{
        headerStyle: {
          backgroundColor: '#1f77b4',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      }}
    >
      <Drawer.Screen 
        name="MainTabs" 
        component={MainTabNavigator}
        options={{ 
          title: 'Food Vision Pro',
          headerShown: false,
        }}
      />
      <Drawer.Screen 
        name="Settings" 
        component={SettingsScreen}
        options={{ 
          title: '⚙️ Settings',
          drawerIcon: ({ color, size }) => (
            <TabBarIcon route={{ name: 'Settings' }} focused={false} color={color} size={size} />
          ),
        }}
      />
    </Drawer.Navigator>
  );
}

// Main app component
export default function App() {
  const { isAuthenticated } = useAuthStore();

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <PaperProvider>
          <NavigationContainer>
            <StatusBar style="auto" />
            
            {isAuthenticated ? (
              <Stack.Navigator screenOptions={{ headerShown: false }}>
                <Stack.Screen name="Main" component={MainDrawerNavigator} />
                <Stack.Screen name="Review" component={ReviewScreen} />
              </Stack.Navigator>
            ) : (
              <Stack.Navigator screenOptions={{ headerShown: false }}>
                <Stack.Screen name="Login" component={LoginScreen} />
              </Stack.Navigator>
            )}
          </NavigationContainer>
        </PaperProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
