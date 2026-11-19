# BlindChat

A React chat demo with Google sign-in and live Firestore updates. Signed-in users share one room.

1. Use a Firebase project created for this demo. Enable Google sign-in and create a Firestore database.
2. Apply the included `firestore.rules` to that database using the Firebase console or run `firebase deploy --only firestore:rules --project YOUR_DEMO_PROJECT_ID`.
3. Copy `.env.example` to `.env.local` and fill in your Firebase web app settings.
4. Run `npm ci` and `npm start` in this directory.

The original dependency versions are preserved, including the legacy `node-sass` dependency. Use an isolated Node 16 environment if installing those versions.
