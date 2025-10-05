export default function RecyclingPage() {
  return (
    <div className="min-h-screen bg-green-50 flex flex-col items-center p-8">
      <h1 className="text-4xl font-bold text-green-700 mb-6">
        ♻️ Recycling Guide
      </h1>

      <p className="text-lg text-gray-700 mb-10">
        Help the environment by sorting waste into the correct bins.
      </p>

      <div className="grid grid-cols-3 gap-8 w-full max-w-4xl">
        <div className="p-6 rounded-2xl shadow-md bg-green-100 text-center">
          <h2 className="text-2xl font-semibold text-green-700">Glass</h2>
          <p>🍾 Bottles, jars</p>
        </div>

        <div className="p-6 rounded-2xl shadow-md bg-yellow-100 text-center">
          <h2 className="text-2xl font-semibold text-yellow-600">Plastic</h2>
          <p>🥤 Bottles, packaging</p>
        </div>

        <div className="p-6 rounded-2xl shadow-md bg-blue-100 text-center">
          <h2 className="text-2xl font-semibold text-blue-600">Paper</h2>
          <p>📰 Newspapers, cardboard</p>
        </div>
      </div>
    </div>
  );
}
